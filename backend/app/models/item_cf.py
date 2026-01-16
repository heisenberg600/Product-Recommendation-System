"""Item-Item Collaborative Filtering Recommender"""

import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings
from app.models.base import BaseRecommender
from app.schemas.recommendation import ModelType, RecommendationItem


class ItemCFRecommender(BaseRecommender):
    """
    Item-Item Collaborative Filtering based on co-purchase patterns.

    This model computes item similarities based on which users bought them together.
    For a given user, it recommends items similar to what they've already purchased.
    """

    model_type = ModelType.ITEM_CF

    def __init__(self, top_k: int = 50, min_support: int = 2):
        """
        Initialize Item-CF recommender.

        Args:
            top_k: Number of similar items to store per item
            min_support: Minimum co-purchases to consider similarity
        """
        super().__init__()
        self.top_k = top_k
        self.min_support = min_support
        self._item_similarities: dict[str, list[tuple[str, float]]] = {}
        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._n_items = 0
        self._n_users = 0

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the Item-CF model by computing item-item similarities.

        Args:
            df: Transaction DataFrame
        """
        start_time = time.time()
        logger.info("Training Item-CF model...")

        # Store item prices
        price_df = df.groupby("item_id")["item_price"].mean()
        self._item_prices = price_df.to_dict()

        # Create user-item matrix
        users = df["user_id"].unique()
        items = df["item_id"].unique()

        self._n_users = len(users)
        self._n_items = len(items)

        logger.info(f"Building matrix: {self._n_users} users x {self._n_items} items")

        # Create mappings
        user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {item: i for i, item in enumerate(items)}
        self._idx_to_item = {i: item for item, i in self._item_to_idx.items()}

        # Build sparse user-item matrix
        rows = []
        cols = []
        data = []

        for _, row in df.iterrows():
            user_idx = user_to_idx[row["user_id"]]
            item_idx = self._item_to_idx[row["item_id"]]
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(1)  # Binary interaction

        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self._n_users, self._n_items)
        )

        # Transpose to get item-user matrix
        item_user_matrix = user_item_matrix.T.tocsr()

        # Compute item-item similarities (using cosine similarity)
        logger.info("Computing item similarities...")

        # Compute similarity in batches to save memory
        batch_size = 1000
        self._item_similarities = {}

        for start_idx in range(0, self._n_items, batch_size):
            end_idx = min(start_idx + batch_size, self._n_items)
            batch_matrix = item_user_matrix[start_idx:end_idx]

            # Compute similarities for this batch against all items
            similarities = cosine_similarity(batch_matrix, item_user_matrix)

            for i, local_idx in enumerate(range(start_idx, end_idx)):
                item_id = self._idx_to_item[local_idx]
                sim_scores = similarities[i]

                # Get top-k similar items (excluding self)
                top_indices = np.argsort(sim_scores)[::-1][:self.top_k + 1]

                similar_items = []
                for idx in top_indices:
                    if idx != local_idx and sim_scores[idx] > 0:
                        similar_items.append((
                            self._idx_to_item[idx],
                            float(sim_scores[idx])
                        ))

                if similar_items:
                    self._item_similarities[item_id] = similar_items[:self.top_k]

        self._is_trained = True
        self._last_trained = datetime.utcnow()
        self._training_time = time.time() - start_time

        logger.info(
            f"Item-CF training complete in {self._training_time:.2f}s. "
            f"Computed similarities for {len(self._item_similarities)} items."
        )

    def recommend(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        n: int = 5,
        exclude_items: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None
    ) -> list[RecommendationItem]:
        """
        Generate recommendations based on user's purchase history.

        Args:
            user_id: User identifier
            user_history: User's purchase history
            n: Number of recommendations
            exclude_items: Items to exclude
            price_range: Price filter

        Returns:
            List of recommendations
        """
        if not self._is_trained:
            logger.warning("Item-CF model not trained")
            return []

        if user_history.empty:
            logger.debug(f"No history for user {user_id}")
            return []

        exclude_items = exclude_items or set()

        # Get user's purchased items
        purchased_items = set(user_history["item_id"].unique())
        exclude_items = exclude_items.union(purchased_items)

        # Aggregate scores from similar items
        candidate_scores: dict[str, float] = defaultdict(float)
        candidate_support: dict[str, int] = defaultdict(int)

        # Weight recent purchases more heavily
        user_history = user_history.sort_values("ticket_datetime", ascending=False)
        recency_weights = np.exp(-np.arange(len(user_history)) * 0.1)

        for idx, (_, row) in enumerate(user_history.iterrows()):
            item_id = row["item_id"]
            weight = recency_weights[min(idx, len(recency_weights) - 1)]

            if item_id in self._item_similarities:
                for similar_item, similarity in self._item_similarities[item_id]:
                    if similar_item not in exclude_items:
                        candidate_scores[similar_item] += similarity * weight
                        candidate_support[similar_item] += 1

        # Filter by price range
        if price_range:
            filtered_candidates = self._filter_by_price(
                list(candidate_scores.keys()),
                price_range
            )
            candidate_scores = {
                k: v for k, v in candidate_scores.items()
                if k in filtered_candidates
            }

        # Sort by score and get top N
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        if not sorted_candidates:
            return []

        # Normalize scores
        max_score = sorted_candidates[0][1]

        recommendations = []
        for item_id, score in sorted_candidates:
            relevance = score / max_score if max_score > 0 else 0

            # Confidence based on support (how many purchased items contributed)
            support = candidate_support[item_id]
            confidence = min(1.0, support / 5)  # Cap at 5 supporting items

            recommendations.append(self._create_recommendation_item(
                item_id=item_id,
                relevance_score=relevance,
                confidence=confidence,
                reason=f"Similar to {support} items in your purchase history"
            ))

        return recommendations

    def get_similar_items(
        self,
        item_id: str,
        n: int = 10
    ) -> list[tuple[str, float]]:
        """
        Get similar items for a given item.

        Args:
            item_id: Item identifier
            n: Number of similar items

        Returns:
            List of (item_id, similarity) tuples
        """
        if item_id not in self._item_similarities:
            return []

        return self._item_similarities[item_id][:n]

    def get_stats(self) -> dict:
        """Get model statistics"""
        base_stats = super().get_stats()
        base_stats.update({
            "n_users": self._n_users,
            "n_items": self._n_items,
            "items_with_similarities": len(self._item_similarities),
            "top_k": self.top_k,
            "min_support": self.min_support
        })
        return base_stats
