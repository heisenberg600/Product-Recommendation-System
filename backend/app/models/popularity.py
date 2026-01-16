"""Popularity-based Recommender for cold-start scenarios"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.models.base import BaseRecommender
from app.schemas.recommendation import ModelType, RecommendationItem


class PopularityRecommender(BaseRecommender):
    """
    Popularity-based recommender using purchase frequency and recency.

    This model recommends globally popular items, with options for
    time-weighted popularity and diversity across price segments.
    """

    model_type = ModelType.POPULARITY

    def __init__(self, time_decay: float = 0.01, diversity_boost: bool = True):
        """
        Initialize Popularity recommender.

        Args:
            time_decay: Decay factor for time-weighting (higher = more recency bias)
            diversity_boost: Whether to boost diversity across price segments
        """
        super().__init__()
        self.time_decay = time_decay
        self.diversity_boost = diversity_boost

        self._item_popularity: dict[str, float] = {}
        self._item_purchase_count: dict[str, int] = {}
        self._item_unique_buyers: dict[str, int] = {}
        self._price_segments: dict[str, list[str]] = {}  # segment -> [item_ids]
        self._trending_items: list[str] = []  # Recent popular items
        self._n_items = 0

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the popularity model by computing item popularity scores.

        Args:
            df: Transaction DataFrame
        """
        start_time = time.time()
        logger.info("Training Popularity model...")

        # Store item prices
        price_df = df.groupby("item_id")["item_price"].mean()
        self._item_prices = price_df.to_dict()

        # Compute basic popularity metrics
        item_stats = df.groupby("item_id").agg({
            "user_id": "nunique",
            "ticket_number": "count",
            "ticket_datetime": "max"
        }).reset_index()
        item_stats.columns = ["item_id", "unique_buyers", "purchase_count", "last_purchase"]

        self._n_items = len(item_stats)

        # Time-weighted popularity
        if not df.empty:
            max_date = df["ticket_datetime"].max()
            item_stats["days_since_last"] = (
                max_date - item_stats["last_purchase"]
            ).dt.days.fillna(0)

            # Popularity score with time decay
            item_stats["time_weight"] = np.exp(
                -self.time_decay * item_stats["days_since_last"]
            )
            item_stats["popularity_score"] = (
                item_stats["purchase_count"] * item_stats["time_weight"]
            )

            # Normalize
            max_pop = item_stats["popularity_score"].max()
            if max_pop > 0:
                item_stats["popularity_score"] = item_stats["popularity_score"] / max_pop

        # Store metrics
        for _, row in item_stats.iterrows():
            item_id = row["item_id"]
            self._item_popularity[item_id] = row.get("popularity_score", 0)
            self._item_purchase_count[item_id] = row["purchase_count"]
            self._item_unique_buyers[item_id] = row["unique_buyers"]

        # Create price segments for diversity
        self._create_price_segments(df)

        # Compute trending items (recent popularity)
        self._compute_trending_items(df)

        self._is_trained = True
        self._last_trained = datetime.utcnow()
        self._training_time = time.time() - start_time

        logger.info(
            f"Popularity training complete in {self._training_time:.2f}s. "
            f"Computed popularity for {len(self._item_popularity)} items."
        )

    def _create_price_segments(self, df: pd.DataFrame) -> None:
        """Create price segments for diversity boosting"""
        if df.empty:
            return

        # Get item prices
        item_prices = df.groupby("item_id")["item_price"].mean()

        # Create segments based on price quartiles
        try:
            segments = pd.qcut(item_prices, q=4, labels=["budget", "mid_low", "mid_high", "premium"])
        except ValueError:
            # If not enough unique values, use simple bins
            segments = pd.cut(
                item_prices,
                bins=[0, 2, 5, 10, float("inf")],
                labels=["budget", "mid_low", "mid_high", "premium"]
            )

        self._price_segments = {}
        for segment in ["budget", "mid_low", "mid_high", "premium"]:
            items = segments[segments == segment].index.tolist()
            if items:
                # Sort by popularity within segment
                items.sort(key=lambda x: self._item_popularity.get(x, 0), reverse=True)
                self._price_segments[segment] = items

    def _compute_trending_items(self, df: pd.DataFrame, days: int = 7) -> None:
        """Compute recently trending items"""
        if df.empty:
            self._trending_items = []
            return

        max_date = df["ticket_datetime"].max()
        recent_cutoff = max_date - pd.Timedelta(days=days)

        recent_df = df[df["ticket_datetime"] >= recent_cutoff]

        if recent_df.empty:
            # Fall back to overall popular
            self._trending_items = sorted(
                self._item_popularity.keys(),
                key=lambda x: self._item_popularity[x],
                reverse=True
            )[:100]
        else:
            trending_counts = recent_df.groupby("item_id").size().sort_values(ascending=False)
            self._trending_items = trending_counts.head(100).index.tolist()

    def recommend(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        n: int = 5,
        exclude_items: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None
    ) -> list[RecommendationItem]:
        """
        Generate popularity-based recommendations.

        Args:
            user_id: User identifier (not used but required for interface)
            user_history: User's purchase history (used for exclusions)
            n: Number of recommendations
            exclude_items: Items to exclude
            price_range: Price filter

        Returns:
            List of recommendations
        """
        if not self._is_trained:
            logger.warning("Popularity model not trained")
            return []

        exclude_items = exclude_items or set()

        # Add purchased items to exclusions
        if not user_history.empty:
            purchased_items = set(user_history["item_id"].unique())
            exclude_items = exclude_items.union(purchased_items)

        # Get candidates
        if self.diversity_boost and not price_range:
            candidates = self._get_diverse_candidates(n * 3, exclude_items)
        else:
            candidates = self._get_top_popular(n * 3, exclude_items)

        # Filter by price range
        if price_range:
            candidates = [
                item_id for item_id in candidates
                if item_id in self._filter_by_price([item_id], price_range)
            ]

        # Take top N
        top_items = candidates[:n]

        if not top_items:
            return []

        recommendations = []
        for rank, item_id in enumerate(top_items):
            popularity = self._item_popularity.get(item_id, 0)
            unique_buyers = self._item_unique_buyers.get(item_id, 0)

            # Confidence based on number of unique buyers
            confidence = min(1.0, unique_buyers / 50)

            # Determine recommendation reason
            if item_id in self._trending_items[:20]:
                reason = f"Trending item - bought by {unique_buyers} customers recently"
            else:
                reason = f"Popular item - purchased by {unique_buyers} customers"

            recommendations.append(self._create_recommendation_item(
                item_id=item_id,
                relevance_score=popularity,
                confidence=confidence,
                reason=reason
            ))

        return recommendations

    def _get_top_popular(
        self,
        n: int,
        exclude_items: set[str]
    ) -> list[str]:
        """Get top N popular items excluding specified items"""
        sorted_items = sorted(
            self._item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = []
        for item_id, _ in sorted_items:
            if item_id not in exclude_items:
                result.append(item_id)
                if len(result) >= n:
                    break

        return result

    def _get_diverse_candidates(
        self,
        n: int,
        exclude_items: set[str]
    ) -> list[str]:
        """Get diverse candidates across price segments"""
        result = []
        segment_idx = defaultdict(int)
        segments = list(self._price_segments.keys())

        if not segments:
            return self._get_top_popular(n, exclude_items)

        # Round-robin from each segment
        while len(result) < n:
            added = False
            for segment in segments:
                items = self._price_segments.get(segment, [])
                idx = segment_idx[segment]

                # Find next valid item from this segment
                while idx < len(items):
                    item_id = items[idx]
                    idx += 1
                    segment_idx[segment] = idx

                    if item_id not in exclude_items and item_id not in result:
                        result.append(item_id)
                        added = True
                        break

            if not added:
                # All segments exhausted
                break

        return result

    def get_trending(self, n: int = 10) -> list[str]:
        """Get trending items"""
        return self._trending_items[:n]

    def get_stats(self) -> dict:
        """Get model statistics"""
        base_stats = super().get_stats()
        base_stats.update({
            "n_items": self._n_items,
            "n_price_segments": len(self._price_segments),
            "n_trending_items": len(self._trending_items),
            "diversity_boost": self.diversity_boost,
            "time_decay": self.time_decay
        })
        return base_stats
