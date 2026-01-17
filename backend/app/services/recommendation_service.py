"""
Recommendation Service with Two-Level Architecture

Level 1: Candidate Generation (200 candidates)
- ALS (40%) + Item-CF (40%) + Popularity (20%) weighted blend

Level 2: Re-ranking
- User segment filtering
- Upsell boost
- Repurchase cycle exclusion
- Final scoring with relevance and confidence
"""

import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
from cachetools import TTLCache
from loguru import logger

from app.core.config import settings
from app.core.tuning_config import TUNING_CONFIG, get_config
from app.models.model_loader import ModelLoader, get_model_loader
from app.models.candidate_ranker import CandidateRanker
from app.schemas.recommendation import (
    ModelType,
    RecommendationItem,
    RecommendationResponse,
    UserInfo,
    UserType,
    SimilarItemsResponse,
    SimilarItem,
)


class RecommendationService:
    """
    Main service for generating product recommendations.

    Uses two-level architecture:
    - Level 1: Generate 200 candidates from model blend
    - Level 2: Re-rank based on user segment, price, and business rules
    """

    def __init__(self, models_path: str = "models"):
        self._models_path = models_path
        self._model_loader: Optional[ModelLoader] = None
        self._candidate_ranker: Optional[CandidateRanker] = None

        # Configuration
        self._config = TUNING_CONFIG

        # Cache for user recommendations
        cache_config = self._config.get('cache', {})
        self._cache: TTLCache = TTLCache(
            maxsize=1000,
            ttl=cache_config.get('ttl_seconds', 3600)
        )

        self._is_initialized = False
        self._model_load_time: Optional[datetime] = None

    async def initialize(self) -> bool:
        """
        Initialize the service by loading pre-trained models.

        Returns:
            True if initialization was successful
        """
        logger.info("Initializing recommendation service...")

        try:
            # Load pre-trained models
            self._model_loader = get_model_loader(self._models_path)

            if not self._model_loader.is_loaded:
                logger.error("Failed to load models")
                return False

            # Initialize candidate ranker
            self._candidate_ranker = CandidateRanker(
                self._config,
                self._model_loader
            )

            self._is_initialized = True
            self._model_load_time = datetime.utcnow()

            # Log model stats
            stats = self._model_loader.get_model_stats()
            logger.info(f"Models loaded - Version: {stats['version']}, "
                       f"Users: {stats['n_users']}, Items: {stats['n_items']}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize recommendation service: {e}")
            return False

    def get_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 5,
        timestamp: Optional[datetime] = None,
        model_type: Optional[str] = None,
        exclude_purchased: bool = True,
        price_range_min: Optional[float] = None,
        price_range_max: Optional[float] = None,
    ) -> RecommendationResponse:
        """
        Generate recommendations for a user using two-level architecture.

        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to return
            timestamp: Current request timestamp (for repurchase cycle)
            model_type: Model to use - 'als', 'item_cf', or 'hybrid' (default)
            exclude_purchased: Whether to exclude already purchased items
            price_range_min: Minimum price filter
            price_range_max: Maximum price filter

        Returns:
            RecommendationResponse with user info and recommendations
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = datetime.now()

        # Check cache
        cache_key = self._get_cache_key(
            user_id, num_recommendations, timestamp, model_type,
            exclude_purchased, price_range_min, price_range_max
        )

        if cache_key in self._cache:
            logger.debug(f"Cache hit for user {user_id}")
            cached_response = self._cache[cache_key]
            cached_response.processing_time_ms = (time.time() - start_time) * 1000
            return cached_response

        # Get user information
        user_info = self._get_user_info(user_id)
        is_new_user = not self._model_loader.is_known_user(user_id)

        # Prepare exclude items
        exclude_items = []
        if exclude_purchased and not is_new_user:
            user_profile = self._model_loader.get_user_profile(user_id)
            if user_profile:
                exclude_items = list(user_profile.get('item_last_purchase', {}).keys())

        # Generate recommendations based on user type
        recommendations: List[RecommendationItem] = []

        # Determine primary model from model_type parameter
        model_type_normalized = (model_type or 'hybrid').lower()
        # Handle 'matrix_factorization' as alias for 'als'
        if model_type_normalized in ('als', 'matrix_factorization'):
            primary_model = ModelType.ALS
            model_type = 'als'  # Normalize for later use
        elif model_type_normalized == 'item_cf':
            primary_model = ModelType.ITEM_CF
        else:
            primary_model = ModelType.HYBRID

        fallback_used = False

        try:
            if is_new_user:
                # New user: Use popularity filtered by price
                recommendations = self._get_new_user_recommendations(
                    user_id=user_id,
                    n=num_recommendations,
                    exclude_items=exclude_items,
                    price_range_min=price_range_min,
                    price_range_max=price_range_max
                )
                primary_model = ModelType.POPULARITY
            else:
                # Existing user: Two-level recommendation
                recommendations = self._get_existing_user_recommendations(
                    user_id=user_id,
                    n=num_recommendations,
                    timestamp=timestamp,
                    model_type=model_type,
                    exclude_items=exclude_items,
                    price_range_min=price_range_min,
                    price_range_max=price_range_max
                )

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")

        # Fallback to popularity if no recommendations
        if not recommendations:
            logger.info(f"Using popularity fallback for user {user_id}")
            fallback_used = True
            primary_model = ModelType.POPULARITY

            popular_items = self._model_loader.get_popular_items(
                n=num_recommendations,
                exclude_items=exclude_items
            )

            for item_id, score, confidence in popular_items:
                item_features = self._model_loader.get_item_features(item_id)
                item_price = item_features.get('avg_price', 0) if item_features else 0

                recommendations.append(RecommendationItem(
                    item_id=item_id,
                    relevance_score=round(score, 4),
                    confidence_score=round(confidence, 4),
                    item_price=round(item_price, 2),
                    recommendation_reason="Popular item",
                    model_source="popularity"
                ))

        processing_time = (time.time() - start_time) * 1000

        response = RecommendationResponse(
            user_id=user_id,
            user_info=user_info,
            recommendations=recommendations,
            primary_model=primary_model,
            fallback_used=fallback_used,
            processing_time_ms=processing_time,
            generated_at=datetime.utcnow()
        )

        # Cache the response
        self._cache[cache_key] = response

        logger.info(
            f"Generated {len(recommendations)} recommendations for user {user_id} "
            f"in {processing_time:.2f}ms (model: {primary_model.value})"
        )

        return response

    def _get_new_user_recommendations(
        self,
        user_id: str,
        n: int,
        exclude_items: List[str],
        price_range_min: Optional[float] = None,
        price_range_max: Optional[float] = None
    ) -> List[RecommendationItem]:
        """
        Get recommendations for a new user using popularity filtered by price.
        """
        # Use candidate ranker's new user method
        ranked_items = self._candidate_ranker.rank_for_new_user(
            first_purchase_price=None,  # No history yet
            n=n,
            exclude_items=exclude_items
        )

        # Apply price filters if specified
        recommendations = []
        for item in ranked_items:
            if price_range_min and item['item_price'] < price_range_min:
                continue
            if price_range_max and item['item_price'] > price_range_max:
                continue

            recommendations.append(RecommendationItem(
                item_id=item['item_id'],
                relevance_score=item['relevance_score'],
                confidence_score=item['confidence_score'],
                item_price=item['item_price'],
                recommendation_reason=item['recommendation_reason'],
                model_source=item['model_source']
            ))

        return recommendations[:n]

    def _get_existing_user_recommendations(
        self,
        user_id: str,
        n: int,
        timestamp: datetime,
        exclude_items: List[str],
        model_type: Optional[str] = None,
        price_range_min: Optional[float] = None,
        price_range_max: Optional[float] = None
    ) -> List[RecommendationItem]:
        """
        Get recommendations for existing user using two-level architecture.

        Level 1: Generate candidates from selected model(s)
        Level 2: Re-rank with business logic

        Args:
            model_type: 'als', 'item_cf', or 'hybrid' (default, uses weighted blend)
        """
        # Get configuration
        candidate_pool_size = self._config.get('candidate_pool_size', 200)

        # =====================================================================
        # LEVEL 1: Candidate Generation
        # =====================================================================

        candidates = []

        # Normalize model_type
        model_type = (model_type or 'hybrid').lower()

        if model_type == 'als':
            # Use only ALS model
            als_recs = self._model_loader.get_als_recommendations(
                user_id, n=candidate_pool_size, exclude_items=exclude_items
            )
            for item_id, score, confidence in als_recs:
                candidates.append((item_id, score, confidence, 'als'))

        elif model_type == 'item_cf':
            # Use only Item-CF model
            item_cf_recs = self._model_loader.get_item_cf_recommendations(
                user_id, n=candidate_pool_size, exclude_items=exclude_items
            )
            for item_id, score, confidence in item_cf_recs:
                candidates.append((item_id, score, confidence, 'item_cf'))

        else:
            # Hybrid: Use weighted blend of all models
            model_weights = self._config.get('model_weights', {
                'als': 0.4, 'item_cf': 0.4, 'popularity': 0.2
            })

            # Get ALS recommendations
            als_weight = model_weights.get('als', 0.4)
            als_n = int(candidate_pool_size * als_weight * 2)
            als_recs = self._model_loader.get_als_recommendations(
                user_id, n=als_n, exclude_items=exclude_items
            )
            for item_id, score, confidence in als_recs:
                candidates.append((item_id, score * als_weight, confidence, 'als'))

            # Get Item-CF recommendations
            item_cf_weight = model_weights.get('item_cf', 0.4)
            item_cf_n = int(candidate_pool_size * item_cf_weight * 2)
            item_cf_recs = self._model_loader.get_item_cf_recommendations(
                user_id, n=item_cf_n, exclude_items=exclude_items
            )
            for item_id, score, confidence in item_cf_recs:
                candidates.append((item_id, score * item_cf_weight, confidence, 'item_cf'))

            # Get Popularity recommendations
            pop_weight = model_weights.get('popularity', 0.2)
            pop_n = int(candidate_pool_size * pop_weight * 2)
            pop_recs = self._model_loader.get_popular_items(
                n=pop_n, exclude_items=exclude_items
            )
            for item_id, score, confidence in pop_recs:
                candidates.append((item_id, score * pop_weight, confidence, 'popularity'))

        # Aggregate scores for items appearing in multiple models (for hybrid)
        item_scores: Dict[str, List[Tuple[float, float, str]]] = {}
        for item_id, score, confidence, source in candidates:
            if item_id not in item_scores:
                item_scores[item_id] = []
            item_scores[item_id].append((score, confidence, source))

        # Combine scores (sum weighted scores, average confidence)
        aggregated_candidates = []
        for item_id, scores_list in item_scores.items():
            total_score = sum(s[0] for s in scores_list)
            avg_confidence = np.mean([s[1] for s in scores_list])

            # Determine primary source
            sources = [s[2] for s in scores_list]
            if len(set(sources)) > 1:
                primary_source = 'hybrid'
            else:
                primary_source = sources[0]

            aggregated_candidates.append((item_id, total_score, avg_confidence, primary_source))

        # Sort by score and take top candidates
        aggregated_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = aggregated_candidates[:candidate_pool_size]

        # =====================================================================
        # LEVEL 2: Re-ranking
        # =====================================================================

        ranked_items = self._candidate_ranker.rank(
            candidates=top_candidates,
            user_id=user_id,
            timestamp=timestamp,
            n=n,
            exclude_items=exclude_items
        )

        # Apply additional price filters if specified
        recommendations = []
        for item in ranked_items:
            if price_range_min and item['item_price'] < price_range_min:
                continue
            if price_range_max and item['item_price'] > price_range_max:
                continue

            recommendations.append(RecommendationItem(
                item_id=item['item_id'],
                relevance_score=item['relevance_score'],
                confidence_score=item['confidence_score'],
                item_price=item['item_price'],
                recommendation_reason=item['recommendation_reason'],
                model_source=item['model_source']
            ))

        return recommendations[:n]

    def get_similar_items(
        self,
        item_id: str,
        n: int = 5
    ) -> SimilarItemsResponse:
        """
        Get items similar to a given item.

        Args:
            item_id: Item identifier
            n: Number of similar items

        Returns:
            SimilarItemsResponse with similar items
        """
        start_time = time.time()

        similar_items = []

        # Get similar items from Item-CF model
        similar_raw = self._model_loader.get_similar_items(item_id, n)

        # Normalize scores to 0-1 range if needed
        if similar_raw:
            max_score = max(score for _, score, _ in similar_raw)
            if max_score > 1.0:
                # Normalize by max score
                similar_raw = [(item, score / max_score, conf) for item, score, conf in similar_raw]

        for similar_item_id, relevance_score, confidence_score in similar_raw:
            item_features = self._model_loader.get_item_features(similar_item_id)
            item_price = item_features.get('avg_price', 0) if item_features else 0

            # Ensure scores are clamped to 0-1 range
            relevance_score = min(max(relevance_score, 0.0), 1.0)
            confidence_score = min(max(confidence_score, 0.0), 1.0)

            similar_items.append(SimilarItem(
                item_id=similar_item_id,
                relevance_score=round(relevance_score, 4),
                confidence_score=round(confidence_score, 4),
                item_price=round(item_price, 2)
            ))

        processing_time = (time.time() - start_time) * 1000

        return SimilarItemsResponse(
            item_id=item_id,
            similar_items=similar_items,
            processing_time_ms=processing_time
        )

    def _get_user_info(self, user_id: str) -> UserInfo:
        """Get user information from loaded models."""
        user_profile = self._model_loader.get_user_profile(user_id)
        user_segment = self._model_loader.get_user_segment(user_id)

        if not user_profile:
            return UserInfo(
                user_id=user_id,
                user_type=UserType.NEW,
                spending_segment=None,
                total_purchases=0,
                unique_items=0,
                avg_item_price=None,
                last_purchase_date=None
            )

        # Get user_type from profile (based on original data source)
        is_loyal = user_profile.get('is_loyal', False)
        if is_loyal:
            user_type = UserType.LOYAL
        else:
            user_type = UserType.NEW

        last_purchase_str = user_profile.get('last_purchase')
        last_purchase_date = None
        if last_purchase_str:
            try:
                last_purchase_date = datetime.fromisoformat(last_purchase_str)
            except (ValueError, TypeError):
                pass

        return UserInfo(
            user_id=user_id,
            user_type=user_type,
            spending_segment=user_segment,
            total_purchases=user_profile.get('total_purchases', 0),
            unique_items=user_profile.get('unique_items', 0),
            avg_item_price=user_profile.get('avg_item_price'),
            last_purchase_date=last_purchase_date
        )

    def _get_cache_key(
        self,
        user_id: str,
        n: int,
        timestamp: datetime,
        model_type: Optional[str],
        exclude_purchased: bool,
        price_min: Optional[float],
        price_max: Optional[float]
    ) -> str:
        """Generate cache key for recommendations."""
        # Round timestamp to nearest hour for caching
        timestamp_key = timestamp.strftime("%Y%m%d%H") if timestamp else "none"
        model_key = model_type or "hybrid"
        return f"{user_id}:{n}:{timestamp_key}:{model_key}:{exclude_purchased}:{price_min}:{price_max}"

    def get_popular_items(self, n: int = 10) -> List[Dict]:
        """
        Get popular items.

        Args:
            n: Number of items

        Returns:
            List of popular items with info
        """
        popular_items = self._model_loader.get_popular_items(n=n)

        result = []
        for item_id, score, confidence in popular_items:
            item_features = self._model_loader.get_item_features(item_id)

            result.append({
                "item_id": item_id,
                "item_price": item_features.get('avg_price') if item_features else None,
                "popularity_score": score,
                "confidence_score": confidence,
                "purchase_count": item_features.get('purchase_count') if item_features else None,
                "unique_buyers": item_features.get('unique_buyers') if item_features else None
            })

        return result

    def get_statistics(self) -> Dict:
        """Get service statistics."""
        model_stats = self._model_loader.get_model_stats() if self._model_loader else {}

        return {
            "is_initialized": self._is_initialized,
            "model_load_time": self._model_load_time.isoformat() if self._model_load_time else None,
            "cache_size": len(self._cache),
            "cache_max_size": self._cache.maxsize,
            "model_stats": model_stats,
            "config": {
                "candidate_pool_size": self._config.get('candidate_pool_size'),
                "model_weights": self._config.get('model_weights'),
                "upsell_enabled": self._config.get('upsell', {}).get('enabled'),
                "repurchase_cycle_enabled": self._config.get('repurchase_cycle', {}).get('enabled')
            }
        }

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._is_initialized

    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._model_loader is not None and self._model_loader.is_loaded


# Global service instance - use models_dir from settings
recommendation_service = RecommendationService(models_path=str(settings.models_dir))
