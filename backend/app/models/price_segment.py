"""Price Segment-based Recommender for price-sensitive recommendations"""

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.models.base import BaseRecommender
from app.schemas.recommendation import ModelType, RecommendationItem


class PriceSegmentRecommender(BaseRecommender):
    """
    Price Segment-based recommender that matches user's price preferences.

    This model analyzes user's historical price preferences and recommends
    items within their typical price range, weighted by popularity.
    """

    model_type = ModelType.PRICE_SEGMENT

    def __init__(self, n_segments: int = 5):
        """
        Initialize Price Segment recommender.

        Args:
            n_segments: Number of price segments to create
        """
        super().__init__()
        self.n_segments = n_segments

        self._segment_boundaries: list[float] = []
        self._segment_items: dict[int, list[tuple[str, float]]] = {}  # segment -> [(item, popularity)]
        self._item_segments: dict[str, int] = {}  # item -> segment
        self._item_popularity: dict[str, float] = {}
        self._n_items = 0

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the price segment model.

        Args:
            df: Transaction DataFrame
        """
        start_time = time.time()
        logger.info("Training Price Segment model...")

        # Store item prices
        price_df = df.groupby("item_id")["item_price"].mean()
        self._item_prices = price_df.to_dict()

        # Compute item popularity
        item_counts = df.groupby("item_id").size()
        max_count = item_counts.max()
        self._item_popularity = (item_counts / max_count).to_dict()

        self._n_items = len(self._item_prices)

        # Create price segments using quantiles
        prices = list(self._item_prices.values())
        if prices:
            try:
                self._segment_boundaries = list(
                    np.percentile(prices, np.linspace(0, 100, self.n_segments + 1))
                )
            except Exception:
                # Fallback to equal-width bins
                min_price = min(prices)
                max_price = max(prices)
                self._segment_boundaries = list(
                    np.linspace(min_price, max_price, self.n_segments + 1)
                )

        # Assign items to segments
        self._segment_items = {i: [] for i in range(self.n_segments)}

        for item_id, price in self._item_prices.items():
            segment = self._get_segment_for_price(price)
            self._item_segments[item_id] = segment
            popularity = self._item_popularity.get(item_id, 0)
            self._segment_items[segment].append((item_id, popularity))

        # Sort items within each segment by popularity
        for segment in self._segment_items:
            self._segment_items[segment].sort(key=lambda x: x[1], reverse=True)

        self._is_trained = True
        self._last_trained = datetime.utcnow()
        self._training_time = time.time() - start_time

        logger.info(
            f"Price Segment training complete in {self._training_time:.2f}s. "
            f"Created {self.n_segments} segments with {self._n_items} items."
        )

        # Log segment distribution
        for seg, items in self._segment_items.items():
            if seg < len(self._segment_boundaries) - 1:
                low = self._segment_boundaries[seg]
                high = self._segment_boundaries[seg + 1]
                logger.debug(f"Segment {seg} (${low:.2f}-${high:.2f}): {len(items)} items")

    def _get_segment_for_price(self, price: float) -> int:
        """Determine which segment a price belongs to"""
        for i in range(len(self._segment_boundaries) - 1):
            if price <= self._segment_boundaries[i + 1]:
                return i
        return self.n_segments - 1

    def recommend(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        n: int = 5,
        exclude_items: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None
    ) -> list[RecommendationItem]:
        """
        Generate price-segment based recommendations.

        Args:
            user_id: User identifier
            user_history: User's purchase history
            n: Number of recommendations
            exclude_items: Items to exclude
            price_range: Price filter (overrides user preferences)

        Returns:
            List of recommendations
        """
        if not self._is_trained:
            logger.warning("Price Segment model not trained")
            return []

        exclude_items = exclude_items or set()

        # Add purchased items to exclusions
        if not user_history.empty:
            purchased_items = set(user_history["item_id"].unique())
            exclude_items = exclude_items.union(purchased_items)

        # Determine user's preferred price segment
        if price_range:
            # Use explicit price range
            target_segments = self._get_segments_in_range(price_range)
        elif not user_history.empty:
            # Infer from purchase history
            target_segments = self._infer_user_segments(user_history)
        else:
            # Default to middle segments
            target_segments = [self.n_segments // 2]

        # Get candidates from target segments
        candidates = self._get_candidates_from_segments(
            target_segments, exclude_items, n * 2
        )

        # Filter by exact price range if specified
        if price_range:
            candidates = [
                (item_id, score) for item_id, score in candidates
                if item_id in self._filter_by_price([item_id], price_range)
            ]

        # Take top N
        top_candidates = candidates[:n]

        if not top_candidates:
            return []

        recommendations = []
        for item_id, popularity in top_candidates:
            segment = self._item_segments.get(item_id, 0)
            price = self._item_prices.get(item_id, 0)

            # Confidence based on how well price matches user's history
            if not user_history.empty:
                user_avg_price = user_history["item_price"].mean()
                price_diff = abs(price - user_avg_price) / (user_avg_price + 1)
                confidence = max(0.3, 1 - price_diff)
            else:
                confidence = 0.5

            # Determine price segment name
            if segment == 0:
                segment_name = "budget-friendly"
            elif segment == self.n_segments - 1:
                segment_name = "premium"
            else:
                segment_name = "mid-range"

            recommendations.append(self._create_recommendation_item(
                item_id=item_id,
                relevance_score=popularity,
                confidence=confidence,
                reason=f"Popular {segment_name} item matching your price preferences"
            ))

        return recommendations

    def _infer_user_segments(self, user_history: pd.DataFrame) -> list[int]:
        """Infer user's preferred price segments from history"""
        if user_history.empty:
            return [self.n_segments // 2]

        # Get distribution of user's purchases across segments
        segment_counts = {}
        for _, row in user_history.iterrows():
            price = row.get("item_price", 0)
            segment = self._get_segment_for_price(price)
            segment_counts[segment] = segment_counts.get(segment, 0) + 1

        if not segment_counts:
            return [self.n_segments // 2]

        # Sort segments by frequency and take top 2
        sorted_segments = sorted(
            segment_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Include adjacent segments for diversity
        result = set()
        for segment, _ in sorted_segments[:2]:
            result.add(segment)
            if segment > 0:
                result.add(segment - 1)
            if segment < self.n_segments - 1:
                result.add(segment + 1)

        return list(result)

    def _get_segments_in_range(
        self,
        price_range: tuple[float, float]
    ) -> list[int]:
        """Get segments that overlap with the price range"""
        min_price, max_price = price_range
        segments = []

        for i in range(len(self._segment_boundaries) - 1):
            seg_low = self._segment_boundaries[i]
            seg_high = self._segment_boundaries[i + 1]

            # Check for overlap
            if seg_low <= max_price and seg_high >= min_price:
                segments.append(i)

        return segments if segments else [self.n_segments // 2]

    def _get_candidates_from_segments(
        self,
        segments: list[int],
        exclude_items: set[str],
        n: int
    ) -> list[tuple[str, float]]:
        """Get top candidates from specified segments"""
        candidates = []

        for segment in segments:
            items = self._segment_items.get(segment, [])
            for item_id, popularity in items:
                if item_id not in exclude_items:
                    candidates.append((item_id, popularity))

        # Sort by popularity and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def get_stats(self) -> dict:
        """Get model statistics"""
        base_stats = super().get_stats()

        segment_sizes = {
            f"segment_{i}": len(items)
            for i, items in self._segment_items.items()
        }

        base_stats.update({
            "n_items": self._n_items,
            "n_segments": self.n_segments,
            "segment_boundaries": self._segment_boundaries,
            "segment_sizes": segment_sizes
        })
        return base_stats
