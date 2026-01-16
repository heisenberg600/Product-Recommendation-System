"""Main recommendation service that orchestrates data loading and model inference"""

import time
from datetime import datetime
from typing import Optional

import pandas as pd
from cachetools import TTLCache
from loguru import logger

from app.core.config import settings
from app.models import (
    HybridRecommender,
    ItemCFRecommender,
    MatrixFactorizationRecommender,
    PopularityRecommender,
    PriceSegmentRecommender,
)
from app.schemas.recommendation import (
    ModelType,
    RecommendationItem,
    RecommendationResponse,
    UserInfo,
    UserType,
)
from app.utils.data_loader import DataLoader


class RecommendationService:
    """
    Main service for generating product recommendations.

    Handles data loading, model training, and inference with caching.
    """

    def __init__(self):
        self._data_loader = DataLoader()
        self._hybrid_model = HybridRecommender()

        # Individual models for direct access
        self._models: dict[ModelType, object] = {
            ModelType.ITEM_CF: ItemCFRecommender(),
            ModelType.MATRIX_FACTORIZATION: MatrixFactorizationRecommender(),
            ModelType.POPULARITY: PopularityRecommender(),
            ModelType.PRICE_SEGMENT: PriceSegmentRecommender(),
            ModelType.HYBRID: self._hybrid_model,
        }

        # Cache for user recommendations
        self._cache: TTLCache = TTLCache(
            maxsize=1000,
            ttl=settings.cache_ttl_seconds
        )

        self._is_initialized = False
        self._last_training_time: Optional[datetime] = None

    async def initialize(self) -> bool:
        """
        Initialize the service by loading data and training models.

        Returns:
            True if initialization was successful
        """
        logger.info("Initializing recommendation service...")

        # Load data
        if not self._data_loader.load_data():
            logger.error("Failed to load data")
            return False

        # Train models
        await self._train_models()

        self._is_initialized = True
        logger.info("Recommendation service initialized successfully")

        return True

    async def _train_models(self) -> None:
        """Train all recommendation models"""
        logger.info("Training recommendation models...")

        combined_data = self._data_loader.combined_data
        if combined_data is None or combined_data.empty:
            logger.warning("No data available for training")
            return

        # Train hybrid model (which trains all components)
        self._hybrid_model.train(combined_data)

        # Train individual models for direct access
        for model_type, model in self._models.items():
            if model_type != ModelType.HYBRID and hasattr(model, "train"):
                try:
                    model.train(combined_data)
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")

        self._last_training_time = datetime.utcnow()
        logger.info("Model training complete")

    def get_recommendations(
        self,
        user_id: str,
        num_recommendations: int = 5,
        model_type: Optional[ModelType] = None,
        exclude_purchased: bool = True,
        price_range_min: Optional[float] = None,
        price_range_max: Optional[float] = None,
    ) -> RecommendationResponse:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to return
            model_type: Specific model to use (None for hybrid/automatic)
            exclude_purchased: Whether to exclude already purchased items
            price_range_min: Minimum price filter
            price_range_max: Maximum price filter

        Returns:
            RecommendationResponse with user info and recommendations
        """
        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(
            user_id, num_recommendations, model_type,
            exclude_purchased, price_range_min, price_range_max
        )

        if cache_key in self._cache:
            logger.debug(f"Cache hit for user {user_id}")
            cached_response = self._cache[cache_key]
            cached_response.processing_time_ms = (time.time() - start_time) * 1000
            return cached_response

        # Get user information
        user_info = self._get_user_info(user_id)
        user_history = self._data_loader.get_user_history(user_id)

        # Prepare exclude items
        exclude_items = None
        if exclude_purchased and not user_history.empty:
            exclude_items = set(user_history["item_id"].unique())

        # Prepare price range
        price_range = None
        if price_range_min is not None or price_range_max is not None:
            price_range = (
                price_range_min or 0,
                price_range_max or float("inf")
            )

        # Get recommendations
        recommendations: list[RecommendationItem] = []
        primary_model = model_type or ModelType.HYBRID
        fallback_used = False

        try:
            if model_type and model_type in self._models:
                # Use specific model
                model = self._models[model_type]
                if hasattr(model, "recommend"):
                    recommendations = model.recommend(
                        user_id=user_id,
                        user_history=user_history,
                        n=num_recommendations,
                        exclude_items=exclude_items,
                        price_range=price_range
                    )
            else:
                # Use hybrid model with user type detection
                recommendations = self._hybrid_model.recommend(
                    user_id=user_id,
                    user_history=user_history,
                    n=num_recommendations,
                    exclude_items=exclude_items,
                    price_range=price_range,
                    user_type=user_info.user_type if user_info.user_type != UserType.UNKNOWN else None
                )

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")

        # Fallback to popularity if no recommendations
        if not recommendations:
            logger.info(f"Using popularity fallback for user {user_id}")
            fallback_used = True
            primary_model = ModelType.POPULARITY

            popularity_model = self._models.get(ModelType.POPULARITY)
            if popularity_model and hasattr(popularity_model, "recommend"):
                recommendations = popularity_model.recommend(
                    user_id=user_id,
                    user_history=user_history,
                    n=num_recommendations,
                    exclude_items=exclude_items,
                    price_range=price_range
                )

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

    def _get_user_info(self, user_id: str) -> UserInfo:
        """Get user information and classify user type"""
        user_history = self._data_loader.get_user_history(user_id)
        user_type_str = self._data_loader.get_user_type(user_id)

        # Map string to enum
        user_type_map = {
            "loyal": UserType.LOYAL,
            "new": UserType.NEW,
            "unknown": UserType.UNKNOWN
        }
        user_type = user_type_map.get(user_type_str, UserType.UNKNOWN)

        if user_history.empty:
            return UserInfo(
                user_id=user_id,
                user_type=user_type,
                total_purchases=0,
                unique_items=0,
                avg_item_price=None,
                last_purchase_date=None
            )

        return UserInfo(
            user_id=user_id,
            user_type=user_type,
            total_purchases=len(user_history),
            unique_items=user_history["item_id"].nunique(),
            avg_item_price=float(user_history["item_price"].mean()),
            last_purchase_date=user_history["ticket_datetime"].max()
        )

    def _get_cache_key(
        self,
        user_id: str,
        n: int,
        model_type: Optional[ModelType],
        exclude_purchased: bool,
        price_min: Optional[float],
        price_max: Optional[float]
    ) -> str:
        """Generate cache key for recommendations"""
        model_str = model_type.value if model_type else "auto"
        return f"{user_id}:{n}:{model_str}:{exclude_purchased}:{price_min}:{price_max}"

    def get_similar_items(self, item_id: str, n: int = 10) -> list[dict]:
        """
        Get items similar to a given item.

        Args:
            item_id: Item identifier
            n: Number of similar items

        Returns:
            List of similar items with scores
        """
        item_cf = self._models.get(ModelType.ITEM_CF)
        if not item_cf or not hasattr(item_cf, "get_similar_items"):
            return []

        similar = item_cf.get_similar_items(item_id, n)

        result = []
        for similar_item_id, score in similar:
            item_info = self._data_loader.get_item_info(similar_item_id)
            result.append({
                "item_id": similar_item_id,
                "similarity_score": score,
                "item_price": item_info.get("avg_price") if item_info else None,
                "purchase_count": item_info.get("purchase_count") if item_info else None
            })

        return result

    def get_popular_items(self, n: int = 10) -> list[dict]:
        """
        Get popular items.

        Args:
            n: Number of items

        Returns:
            List of popular items with info
        """
        popular_ids = self._data_loader.get_popular_items(n)

        result = []
        for item_id in popular_ids:
            item_info = self._data_loader.get_item_info(item_id)
            if item_info:
                result.append({
                    "item_id": item_id,
                    "item_price": item_info.get("avg_price"),
                    "purchase_count": item_info.get("purchase_count"),
                    "unique_buyers": item_info.get("unique_buyers"),
                    "popularity_score": item_info.get("popularity_score")
                })

        return result

    def get_all_users(self) -> dict[str, list[str]]:
        """Get all user IDs grouped by type"""
        result = {"loyal": [], "new": []}

        if self._data_loader.loyal_customers is not None:
            result["loyal"] = self._data_loader.loyal_customers["user_id"].unique().tolist()

        if self._data_loader.new_customers is not None:
            result["new"] = self._data_loader.new_customers["user_id"].unique().tolist()

        return result

    def get_statistics(self) -> dict:
        """Get service statistics"""
        stats = {
            "is_initialized": self._is_initialized,
            "last_training_time": self._last_training_time.isoformat() if self._last_training_time else None,
            "cache_size": len(self._cache),
            "cache_max_size": self._cache.maxsize,
            "data_stats": self._data_loader.get_statistics(),
            "model_stats": {}
        }

        for model_type, model in self._models.items():
            if hasattr(model, "get_stats"):
                stats["model_stats"][model_type.value] = model.get_stats()

        return stats

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._is_initialized

    @property
    def data_loaded(self) -> bool:
        """Check if data is loaded"""
        return self._data_loader.is_loaded


# Global service instance
recommendation_service = RecommendationService()
