"""Matrix Factorization (ALS) Recommender for implicit feedback"""

import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix

from app.core.config import settings
from app.models.base import BaseRecommender
from app.schemas.recommendation import ModelType, RecommendationItem


class MatrixFactorizationRecommender(BaseRecommender):
    """
    Matrix Factorization recommender using Alternating Least Squares (ALS).

    This model learns latent factors for users and items from implicit feedback
    (purchase data), then predicts user preferences for unseen items.
    """

    model_type = ModelType.MATRIX_FACTORIZATION

    def __init__(
        self,
        n_factors: int = 50,
        n_iterations: int = 15,
        regularization: float = 0.01,
        confidence_scale: float = 40.0
    ):
        """
        Initialize MF recommender.

        Args:
            n_factors: Number of latent factors
            n_iterations: Number of ALS iterations
            regularization: Regularization parameter
            confidence_scale: Scale for confidence weighting
        """
        super().__init__()
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.confidence_scale = confidence_scale

        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._user_to_idx: dict[str, int] = {}
        self._idx_to_user: dict[int, str] = {}
        self._item_to_idx: dict[str, int] = {}
        self._idx_to_item: dict[int, str] = {}
        self._n_users = 0
        self._n_items = 0

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the MF model using ALS algorithm.

        Args:
            df: Transaction DataFrame
        """
        start_time = time.time()
        logger.info("Training Matrix Factorization model...")

        # Store item prices
        price_df = df.groupby("item_id")["item_price"].mean()
        self._item_prices = price_df.to_dict()

        # Create user-item interaction counts
        interaction_counts = df.groupby(["user_id", "item_id"]).size().reset_index(name="count")

        users = interaction_counts["user_id"].unique()
        items = interaction_counts["item_id"].unique()

        self._n_users = len(users)
        self._n_items = len(items)

        logger.info(f"Building matrix: {self._n_users} users x {self._n_items} items")

        # Create mappings
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._idx_to_user = {i: u for u, i in self._user_to_idx.items()}
        self._item_to_idx = {item: i for i, item in enumerate(items)}
        self._idx_to_item = {i: item for item, i in self._item_to_idx.items()}

        # Build sparse interaction matrix with confidence
        rows = []
        cols = []
        data = []

        for _, row in interaction_counts.iterrows():
            user_idx = self._user_to_idx[row["user_id"]]
            item_idx = self._item_to_idx[row["item_id"]]
            # Confidence = 1 + alpha * count
            confidence = 1 + self.confidence_scale * np.log1p(row["count"])
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(confidence)

        confidence_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self._n_users, self._n_items)
        )

        # Binary preference matrix (1 if any interaction, 0 otherwise)
        preference_matrix = (confidence_matrix > 0).astype(np.float32)

        # Initialize factors randomly
        np.random.seed(42)
        self._user_factors = np.random.normal(
            0, 0.1, (self._n_users, self.n_factors)
        ).astype(np.float32)
        self._item_factors = np.random.normal(
            0, 0.1, (self._n_items, self.n_factors)
        ).astype(np.float32)

        # ALS iterations
        logger.info(f"Running {self.n_iterations} ALS iterations...")

        for iteration in range(self.n_iterations):
            # Update user factors
            self._user_factors = self._als_step(
                confidence_matrix,
                preference_matrix,
                self._item_factors,
                self._user_factors,
                self.regularization
            )

            # Update item factors
            self._item_factors = self._als_step(
                confidence_matrix.T.tocsr(),
                preference_matrix.T.tocsr(),
                self._user_factors,
                self._item_factors,
                self.regularization
            )

            if (iteration + 1) % 5 == 0:
                logger.debug(f"Completed iteration {iteration + 1}/{self.n_iterations}")

        self._is_trained = True
        self._last_trained = datetime.utcnow()
        self._training_time = time.time() - start_time

        logger.info(f"MF training complete in {self._training_time:.2f}s")

    def _als_step(
        self,
        confidence: csr_matrix,
        preference: csr_matrix,
        fixed_factors: np.ndarray,
        factors_to_update: np.ndarray,
        regularization: float
    ) -> np.ndarray:
        """
        Single ALS update step.

        Args:
            confidence: Confidence matrix
            preference: Preference matrix
            fixed_factors: Factors held fixed (items when updating users, vice versa)
            factors_to_update: Factors to update
            regularization: Regularization parameter

        Returns:
            Updated factors
        """
        n_rows = factors_to_update.shape[0]
        n_factors = fixed_factors.shape[1]

        # Precompute YtY
        YtY = fixed_factors.T @ fixed_factors

        updated_factors = np.zeros_like(factors_to_update)

        for i in range(n_rows):
            # Get confidence and preference for this row
            conf_row = np.array(confidence[i].todense()).flatten()
            pref_row = np.array(preference[i].todense()).flatten()

            # Items with interactions
            nonzero_idx = conf_row > 0

            if nonzero_idx.sum() == 0:
                # No interactions, use regularized average
                updated_factors[i] = np.zeros(n_factors)
                continue

            # Compute (YtCuY + lambda*I)^-1 * YtCu*p(u)
            # Simplified: only consider non-zero entries
            Y_nz = fixed_factors[nonzero_idx]
            C_nz = conf_row[nonzero_idx]
            p_nz = pref_row[nonzero_idx]

            # YtCuY
            YtCY = Y_nz.T @ (Y_nz * C_nz[:, np.newaxis])

            # Add regularization
            A = YtCY + regularization * np.eye(n_factors)

            # YtCu*p(u)
            b = Y_nz.T @ (C_nz * p_nz)

            # Solve for factors
            try:
                updated_factors[i] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                updated_factors[i] = np.zeros(n_factors)

        return updated_factors

    def recommend(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        n: int = 5,
        exclude_items: Optional[set[str]] = None,
        price_range: Optional[tuple[float, float]] = None
    ) -> list[RecommendationItem]:
        """
        Generate recommendations using learned latent factors.

        Args:
            user_id: User identifier
            user_history: User's purchase history (used for exclusions)
            n: Number of recommendations
            exclude_items: Items to exclude
            price_range: Price filter

        Returns:
            List of recommendations
        """
        if not self._is_trained:
            logger.warning("MF model not trained")
            return []

        if user_id not in self._user_to_idx:
            logger.debug(f"User {user_id} not in training data")
            return []

        exclude_items = exclude_items or set()

        # Get user's purchased items
        if not user_history.empty:
            purchased_items = set(user_history["item_id"].unique())
            exclude_items = exclude_items.union(purchased_items)

        # Get user factors
        user_idx = self._user_to_idx[user_id]
        user_factor = self._user_factors[user_idx]

        # Compute scores for all items
        scores = self._item_factors @ user_factor

        # Create candidate list
        candidates = []
        for item_idx, score in enumerate(scores):
            item_id = self._idx_to_item[item_idx]
            if item_id not in exclude_items:
                candidates.append((item_id, float(score)))

        # Filter by price range
        if price_range:
            candidates = [
                (item_id, score) for item_id, score in candidates
                if item_id in self._filter_by_price([item_id], price_range)
            ]

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:n]

        if not top_candidates:
            return []

        # Normalize scores to [0, 1]
        max_score = max(score for _, score in top_candidates)
        min_score = min(score for _, score in top_candidates)
        score_range = max_score - min_score if max_score > min_score else 1

        recommendations = []
        for item_id, score in top_candidates:
            # Normalize score
            normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5

            # Confidence based on user history size
            history_size = len(user_history) if not user_history.empty else 0
            confidence = min(1.0, history_size / 100)  # More history = more confidence

            recommendations.append(self._create_recommendation_item(
                item_id=item_id,
                relevance_score=normalized_score,
                confidence=confidence,
                reason="Based on your preference patterns and similar users"
            ))

        return recommendations

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get the latent factor vector for a user"""
        if user_id not in self._user_to_idx:
            return None
        return self._user_factors[self._user_to_idx[user_id]]

    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get the latent factor vector for an item"""
        if item_id not in self._item_to_idx:
            return None
        return self._item_factors[self._item_to_idx[item_id]]

    def get_stats(self) -> dict:
        """Get model statistics"""
        base_stats = super().get_stats()
        base_stats.update({
            "n_users": self._n_users,
            "n_items": self._n_items,
            "n_factors": self.n_factors,
            "n_iterations": self.n_iterations,
            "regularization": self.regularization
        })
        return base_stats
