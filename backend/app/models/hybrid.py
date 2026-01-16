"""Hybrid Recommender that combines multiple recommendation strategies"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.models.base import BaseRecommender
from app.models.item_cf import ItemCFRecommender
from app.models.matrix_factorization import MatrixFactorizationRecommender
from app.models.popularity import PopularityRecommender
from app.models.price_segment import PriceSegmentRecommender
from app.schemas.recommendation import ModelType, RecommendationItem, UserType


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender that intelligently combines multiple models.

    Strategy:
    - For loyal users: Weight towards Item-CF and MF (personalized)
    - For new users: Weight towards Popularity and Price Segment (cold-start)
    - Dynamically adjusts weights based on user history size
    """

    model_type = ModelType.HYBRID

    def __init__(
        self,
        loyal_weights: Optional[dict[str, float]] = None,
        new_weights: Optional[dict[str, float]] = None,
        min_history_for_cf: int = 5
    ):
        """
        Initialize Hybrid recommender.

        Args:
            loyal_weights: Model weights for loyal users
            new_weights: Model weights for new users
            min_history_for_cf: Minimum purchases to use CF models
        """
        super().__init__()

        # Default weights for loyal users (personalized focus)
        self.loyal_weights = loyal_weights or {
            "item_cf": 0.45,
            "matrix_factorization": 0.35,
            "popularity": 0.10,
            "price_segment": 0.10
        }

        # Default weights for new users (cold-start focus)
        self.new_weights = new_weights or {
            "item_cf": 0.10,
            "matrix_factorization": 0.10,
            "popularity": 0.50,
            "price_segment": 0.30
        }

        self.min_history_for_cf = min_history_for_cf

        # Component models
        self._item_cf = ItemCFRecommender()
        self._mf = MatrixFactorizationRecommender()
        self._popularity = PopularityRecommender()
        self._price_segment = PriceSegmentRecommender()

        self._models = {
            "item_cf": self._item_cf,
            "matrix_factorization": self._mf,
            "popularity": self._popularity,
            "price_segment": self._price_segment
        }

    def train(self, df: pd.DataFrame) -> None:
        """
        Train all component models.

        Args:
            df: Transaction DataFrame
        """
        start_time = time.time()
        logger.info("Training Hybrid model (all components)...")

        # Store item prices from all models
        price_df = df.groupby("item_id")["item_price"].mean()
        self._item_prices = price_df.to_dict()

        # Train each component model
        for name, model in self._models.items():
            logger.info(f"Training {name} component...")
            try:
                model.train(df)
                logger.info(f"{name} training complete")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")

        self._is_trained = True
        self._last_trained = datetime.utcnow()
        self._training_time = time.time() - start_time

        logger.info(f"Hybrid training complete in {self._training_time:.2f}s")

    def recommend(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        n: int = 5,
        exclude_items: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None,
        user_type: Optional[UserType] = None
    ) -> list[RecommendationItem]:
        """
        Generate hybrid recommendations by combining multiple models.

        Args:
            user_id: User identifier
            user_history: User's purchase history
            n: Number of recommendations
            exclude_items: Items to exclude
            price_range: Price filter
            user_type: Force specific user type (loyal/new)

        Returns:
            List of recommendations
        """
        if not self._is_trained:
            logger.warning("Hybrid model not trained")
            return []

        exclude_items = exclude_items or set()

        # Add purchased items to exclusions
        if not user_history.empty:
            purchased_items = set(user_history["item_id"].unique())
            exclude_items = exclude_items.union(purchased_items)

        # Determine user type and weights
        history_size = len(user_history) if not user_history.empty else 0

        if user_type == UserType.LOYAL or (
            user_type is None and history_size >= self.min_history_for_cf
        ):
            weights = self._get_dynamic_weights(history_size, is_loyal=True)
            effective_user_type = UserType.LOYAL
        else:
            weights = self._get_dynamic_weights(history_size, is_loyal=False)
            effective_user_type = UserType.NEW

        logger.debug(f"User {user_id}: type={effective_user_type}, history={history_size}, weights={weights}")

        # Collect recommendations from all models
        all_recommendations: dict[str, list[RecommendationItem]] = {}

        for model_name, model in self._models.items():
            if weights.get(model_name, 0) > 0 and model.is_trained:
                try:
                    recs = model.recommend(
                        user_id=user_id,
                        user_history=user_history,
                        n=n * 2,  # Get more to allow for filtering
                        exclude_items=exclude_items,
                        price_range=price_range
                    )
                    all_recommendations[model_name] = recs
                except Exception as e:
                    logger.error(f"Error getting recommendations from {model_name}: {e}")

        # Combine recommendations with weighted scoring
        combined_scores: dict[str, float] = defaultdict(float)
        combined_confidence: dict[str, float] = defaultdict(float)
        item_reasons: dict[str, list[str]] = defaultdict(list)
        item_models: dict[str, list[str]] = defaultdict(list)

        for model_name, recs in all_recommendations.items():
            weight = weights.get(model_name, 0)

            for rec in recs:
                item_id = rec.item_id
                combined_scores[item_id] += rec.relevance_score * weight
                combined_confidence[item_id] += rec.confidence * weight
                item_reasons[item_id].append(rec.recommendation_reason)
                item_models[item_id].append(model_name)

        # Filter out excluded items
        combined_scores = {
            k: v for k, v in combined_scores.items()
            if k not in exclude_items
        }

        # Sort by combined score
        sorted_items = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        if not sorted_items:
            # Fallback to pure popularity
            logger.warning(f"No hybrid recommendations for user {user_id}, falling back to popularity")
            return self._popularity.recommend(
                user_id, user_history, n, exclude_items, price_range
            )

        # Normalize scores
        max_score = sorted_items[0][1] if sorted_items else 1

        recommendations = []
        for item_id, score in sorted_items:
            normalized_score = score / max_score if max_score > 0 else 0
            confidence = combined_confidence[item_id]

            # Create combined reason
            models_used = item_models[item_id]
            if len(models_used) > 1:
                reason = f"Recommended by {len(models_used)} models based on your profile"
            else:
                reason = item_reasons[item_id][0] if item_reasons[item_id] else "Based on combined analysis"

            recommendations.append(RecommendationItem(
                item_id=item_id,
                relevance_score=min(1.0, normalized_score),
                confidence=min(1.0, confidence),
                item_price=self._item_prices.get(item_id),
                recommendation_reason=reason,
                model_used=self.model_type
            ))

        return recommendations

    def _get_dynamic_weights(
        self,
        history_size: int,
        is_loyal: bool
    ) -> dict[str, float]:
        """
        Get dynamic weights based on user history size.

        Gradually shifts from cold-start to personalized as history grows.
        """
        if is_loyal:
            base_weights = self.loyal_weights.copy()
        else:
            base_weights = self.new_weights.copy()

        # For users with some but not much history, blend weights
        if 0 < history_size < self.min_history_for_cf:
            blend_factor = history_size / self.min_history_for_cf

            blended = {}
            for model in base_weights:
                loyal_w = self.loyal_weights.get(model, 0)
                new_w = self.new_weights.get(model, 0)
                blended[model] = new_w + (loyal_w - new_w) * blend_factor

            return blended

        return base_weights

    def get_model(self, model_type: ModelType) -> Optional[BaseRecommender]:
        """Get a specific component model"""
        model_map = {
            ModelType.ITEM_CF: self._item_cf,
            ModelType.MATRIX_FACTORIZATION: self._mf,
            ModelType.POPULARITY: self._popularity,
            ModelType.PRICE_SEGMENT: self._price_segment
        }
        return model_map.get(model_type)

    def get_stats(self) -> dict:
        """Get model statistics"""
        base_stats = super().get_stats()

        component_stats = {}
        for name, model in self._models.items():
            component_stats[name] = model.get_stats()

        base_stats.update({
            "loyal_weights": self.loyal_weights,
            "new_weights": self.new_weights,
            "min_history_for_cf": self.min_history_for_cf,
            "component_models": component_stats
        })
        return base_stats
