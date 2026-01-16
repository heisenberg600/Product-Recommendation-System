"""Base class for all recommendation models"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from app.schemas.recommendation import ModelType, RecommendationItem


class BaseRecommender(ABC):
    """Abstract base class for recommendation models"""

    model_type: ModelType

    def __init__(self):
        self._is_trained = False
        self._last_trained: Optional[datetime] = None
        self._training_time: Optional[float] = None
        self._item_prices: dict[str, float] = {}

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the model on transaction data.

        Args:
            df: Transaction DataFrame with columns:
                - user_id: User identifier
                - item_id: Item identifier
                - units_sold: Number of units (optional)
                - item_price: Price of item (optional)
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        n: int = 5,
        exclude_items: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None
    ) -> list[RecommendationItem]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User identifier
            user_history: User's purchase history DataFrame
            n: Number of recommendations
            exclude_items: Items to exclude from recommendations
            price_range: (min_price, max_price) filter

        Returns:
            List of RecommendationItem objects
        """
        pass

    @property
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained

    @property
    def last_trained(self) -> Optional[datetime]:
        """Get last training timestamp"""
        return self._last_trained

    @property
    def training_time(self) -> Optional[float]:
        """Get training time in seconds"""
        return self._training_time

    def set_item_prices(self, prices: dict[str, float]) -> None:
        """Set item prices for including in recommendations"""
        self._item_prices = prices

    def get_item_price(self, item_id: str) -> Optional[float]:
        """Get price for an item"""
        return self._item_prices.get(item_id)

    def _filter_by_price(
        self,
        items: list[str],
        price_range: Optional[tuple[float, float]]
    ) -> list[str]:
        """Filter items by price range"""
        if price_range is None:
            return items

        min_price, max_price = price_range
        filtered = []

        for item_id in items:
            price = self._item_prices.get(item_id)
            if price is None:
                continue
            if min_price is not None and price < min_price:
                continue
            if max_price is not None and price > max_price:
                continue
            filtered.append(item_id)

        return filtered

    def _create_recommendation_item(
        self,
        item_id: str,
        relevance_score: float,
        confidence: float,
        reason: str
    ) -> RecommendationItem:
        """Create a RecommendationItem with proper formatting"""
        return RecommendationItem(
            item_id=item_id,
            relevance_score=min(1.0, max(0.0, relevance_score)),
            confidence=min(1.0, max(0.0, confidence)),
            item_price=self._item_prices.get(item_id),
            recommendation_reason=reason,
            model_used=self.model_type
        )

    def get_stats(self) -> dict:
        """Get model statistics"""
        return {
            "model_type": self.model_type.value,
            "is_trained": self._is_trained,
            "last_trained": self._last_trained.isoformat() if self._last_trained else None,
            "training_time_seconds": self._training_time
        }
