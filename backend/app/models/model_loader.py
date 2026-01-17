"""
Model Loader for Product Recommendation System

Loads pre-trained models from disk on backend startup.
Provides a unified interface for model inference.

Usage:
    from backend.app.models.model_loader import ModelLoader

    # Load all models
    loader = ModelLoader(models_path="models")
    loader.load_all()

    # Get recommendations
    candidates = loader.get_als_recommendations(user_id, n=100)
    similar_items = loader.get_similar_items(item_id, n=5)
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and provides inference for pre-trained recommendation models.

    Models loaded:
    - ALS (Matrix Factorization)
    - Item-Item Collaborative Filtering
    - Popularity Model
    - User Segments
    - User Profiles
    - Item Features
    - Repurchase Cycles
    """

    def __init__(self, models_path: str = "models"):
        self.models_path = Path(models_path)
        self.is_loaded = False

        # Model containers
        self.als_model: Optional[Dict] = None
        self.item_cf_model: Optional[Dict] = None
        self.popularity_model: Optional[Dict] = None
        self.user_segments: Optional[Dict] = None
        self.user_profiles: Optional[Dict] = None
        self.item_features: Optional[Dict] = None
        self.repurchase_cycles: Optional[Dict] = None
        self.metadata: Optional[Dict] = None

    def load_all(self) -> bool:
        """
        Load all models from disk.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info(f"Loading models from {self.models_path}...")

            # Check if models directory exists
            if not self.models_path.exists():
                logger.error(f"Models directory not found: {self.models_path}")
                return False

            # Load metadata first
            metadata_path = self.models_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata - trained at: {self.metadata.get('trained_at')}")

            # Load all models
            self.als_model = self._load_pickle('als_model.pkl')
            self.item_cf_model = self._load_pickle('item_cf_model.pkl')
            self.popularity_model = self._load_pickle('popularity_model.pkl')
            self.user_segments = self._load_pickle('user_segments.pkl')
            self.user_profiles = self._load_pickle('user_profiles.pkl')
            self.item_features = self._load_pickle('item_features.pkl')
            self.repurchase_cycles = self._load_pickle('repurchase_cycles.pkl')

            self.is_loaded = True
            logger.info("All models loaded successfully!")

            # Log summary
            if self.metadata:
                logger.info(f"  Users: {self.metadata.get('n_users', 'N/A')}")
                logger.info(f"  Items: {self.metadata.get('n_items', 'N/A')}")
                logger.info(f"  Version: {self.metadata.get('version', 'N/A')}")

            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.is_loaded = False
            return False

    def _load_pickle(self, filename: str) -> Optional[Dict]:
        """Load a pickle file from the models directory."""
        file_path = self.models_path / filename
        if not file_path.exists():
            logger.warning(f"Model file not found: {file_path}")
            return None

        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"  Loaded {filename}")
        return model

    # =========================================================================
    # ALS Model Methods
    # =========================================================================

    def get_als_recommendations(
        self,
        user_id: str,
        n: int = 100,
        exclude_items: List[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Get top-N recommendations from ALS model.

        Returns:
            List of (item_id, relevance_score, confidence_score) tuples
        """
        if not self.als_model:
            return []

        exclude_items = exclude_items or []

        # Check if user exists
        user_idx = self.als_model['user_to_idx'].get(user_id)
        if user_idx is None:
            logger.debug(f"User {user_id} not in ALS model")
            return []

        # Get user factor
        user_factor = self.als_model['user_factors'][user_idx]

        # Compute scores for all items
        item_factors = self.als_model['item_factors']
        scores = np.dot(item_factors, user_factor)

        # Get user confidence
        user_confidence = float(self.als_model['user_confidence'][user_idx])

        # Create (item_id, score) tuples
        results = []
        for item_idx, score in enumerate(scores):
            item_id = self.als_model['idx_to_item'][item_idx]
            if item_id not in exclude_items:
                # Normalize score to 0-1 range
                normalized_score = self._sigmoid(score)
                results.append((item_id, normalized_score, user_confidence))

        # Sort by score and return top N
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def get_als_user_factor(self, user_id: str) -> Optional[np.ndarray]:
        """Get the latent factor vector for a user."""
        if not self.als_model:
            return None

        user_idx = self.als_model['user_to_idx'].get(user_id)
        if user_idx is None:
            return None

        return self.als_model['user_factors'][user_idx]

    def get_als_item_factor(self, item_id: str) -> Optional[np.ndarray]:
        """Get the latent factor vector for an item."""
        if not self.als_model:
            return None

        item_idx = self.als_model['item_to_idx'].get(item_id)
        if item_idx is None:
            return None

        return self.als_model['item_factors'][item_idx]

    # =========================================================================
    # Item-CF Model Methods
    # =========================================================================

    def get_item_cf_recommendations(
        self,
        user_id: str,
        n: int = 100,
        exclude_items: List[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Get recommendations based on items the user has purchased.
        Uses Item-Item CF to find similar items.

        Returns:
            List of (item_id, relevance_score, confidence_score) tuples
        """
        if not self.item_cf_model or not self.user_profiles:
            return []

        exclude_items = exclude_items or []

        # Get user's purchase history
        user_profile = self.user_profiles.get(user_id)
        if not user_profile:
            return []

        user_items = list(user_profile.get('item_last_purchase', {}).keys())
        if not user_items:
            return []

        # Aggregate similar items from user's history
        item_scores = {}
        similarities = self.item_cf_model['similarities']

        for purchased_item in user_items:
            if purchased_item not in similarities:
                continue

            for similar_item, similarity in similarities[purchased_item]:
                if similar_item in user_items or similar_item in exclude_items:
                    continue

                if similar_item not in item_scores:
                    item_scores[similar_item] = []
                item_scores[similar_item].append(similarity)

        # Compute final scores (average of similarities)
        results = []
        for item_id, scores_list in item_scores.items():
            avg_score = np.mean(scores_list)
            # Confidence based on how many of user's items are similar
            confidence = min(len(scores_list) / 5, 1.0)
            item_confidence = self.item_cf_model['item_confidence'].get(item_id, 0.5)
            combined_confidence = (confidence + item_confidence) / 2
            results.append((item_id, avg_score, combined_confidence))

        # Sort by score and return top N
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def get_similar_items(
        self,
        item_id: str,
        n: int = 5
    ) -> List[Tuple[str, float, float]]:
        """
        Get similar items for a given item.

        Returns:
            List of (item_id, relevance_score, confidence_score) tuples
        """
        if not self.item_cf_model:
            return []

        similarities = self.item_cf_model['similarities'].get(item_id, [])

        results = []
        for similar_item, similarity in similarities[:n]:
            confidence = self.item_cf_model['item_confidence'].get(similar_item, 0.5)
            results.append((similar_item, similarity, confidence))

        return results

    # =========================================================================
    # Popularity Model Methods
    # =========================================================================

    def get_popular_items(
        self,
        n: int = 100,
        exclude_items: List[str] = None,
        price_min: float = None,
        price_max: float = None
    ) -> List[Tuple[str, float, float]]:
        """
        Get popular items, optionally filtered by price range.

        Returns:
            List of (item_id, relevance_score, confidence_score) tuples
        """
        if not self.popularity_model:
            return []

        exclude_items = exclude_items or []

        results = []
        for item_id in self.popularity_model['sorted_items']:
            if item_id in exclude_items:
                continue

            scores = self.popularity_model['scores'].get(item_id, {})
            item_price = scores.get('avg_price', 0) or 0  # Handle None

            # Apply price filters (ensure item_price is a number)
            if price_min is not None and item_price is not None and item_price < price_min:
                continue
            if price_max is not None and item_price is not None and item_price > price_max:
                continue

            popularity_score = scores.get('popularity_score', 0)
            # Confidence is high for popular items
            confidence = min(popularity_score + 0.5, 1.0)

            results.append((item_id, popularity_score, confidence))

            if len(results) >= n:
                break

        return results

    def get_popularity_score(self, item_id: str) -> float:
        """Get the popularity score for an item."""
        if not self.popularity_model:
            return 0.0

        scores = self.popularity_model['scores'].get(item_id, {})
        return scores.get('popularity_score', 0.0)

    # =========================================================================
    # User Methods
    # =========================================================================

    def get_user_segment(self, user_id: str) -> str:
        """Get the spending segment for a user."""
        if not self.user_segments:
            return 'average'

        return self.user_segments['segments'].get(user_id, 'average')

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get the full profile for a user."""
        if not self.user_profiles:
            return None

        return self.user_profiles.get(user_id)

    def get_user_avg_price(self, user_id: str) -> float:
        """Get the average item price for a user."""
        profile = self.get_user_profile(user_id)
        if profile:
            return profile.get('avg_item_price', 0.0)
        return 0.0

    def is_known_user(self, user_id: str) -> bool:
        """Check if a user is known (has history)."""
        if not self.user_profiles:
            return False
        return user_id in self.user_profiles

    def get_user_item_last_purchase(
        self,
        user_id: str,
        item_id: str
    ) -> Optional[datetime]:
        """Get the last purchase date for a user-item pair."""
        profile = self.get_user_profile(user_id)
        if not profile:
            return None

        item_purchases = profile.get('item_last_purchase', {})
        date_str = item_purchases.get(item_id)

        if date_str:
            return datetime.fromisoformat(date_str)
        return None

    # =========================================================================
    # Item Methods
    # =========================================================================

    def get_item_features(self, item_id: str) -> Optional[Dict]:
        """Get features for an item."""
        if not self.item_features:
            return None

        return self.item_features.get(item_id)

    def get_item_price(self, item_id: str) -> float:
        """Get the average price for an item."""
        features = self.get_item_features(item_id)
        if features:
            return features.get('avg_price', 0.0)
        return 0.0

    def get_repurchase_cycle(self, item_id: str) -> Optional[Dict]:
        """Get the repurchase cycle info for an item."""
        if not self.repurchase_cycles:
            return None

        return self.repurchase_cycles.get(item_id)

    def get_avg_repurchase_days(self, item_id: str) -> float:
        """Get the average days between repurchases for an item."""
        cycle = self.get_repurchase_cycle(item_id)
        if cycle:
            return cycle.get('avg_cycle_days', 30.0)
        return 30.0  # Default

    # =========================================================================
    # Segment Thresholds
    # =========================================================================

    def get_segment_price_range(self, segment: str) -> Tuple[float, float]:
        """
        Get the price range for a spending segment.

        Returns:
            (min_price, max_price) tuple
        """
        if not self.user_segments:
            return (0, float('inf'))

        thresholds = self.user_segments.get('thresholds', {})
        p25 = thresholds.get('p25', 0)
        p50 = thresholds.get('p50', 0)
        p75 = thresholds.get('p75', 0)

        ranges = {
            'small': (0, p25),
            'low_average': (p25, p50),
            'average': (p50, p75),
            'high': (p75, float('inf'))
        }

        return ranges.get(segment, (0, float('inf')))

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid to normalize score to 0-1."""
        return 1 / (1 + np.exp(-x))

    def get_model_version(self) -> str:
        """Get the version of loaded models."""
        if self.metadata:
            return self.metadata.get('version', 'unknown')
        return 'unknown'

    def get_training_timestamp(self) -> Optional[str]:
        """Get when the models were trained."""
        if self.metadata:
            return self.metadata.get('trained_at')
        return None

    def get_model_stats(self) -> Dict:
        """Get statistics about loaded models."""
        return {
            'is_loaded': self.is_loaded,
            'version': self.get_model_version(),
            'trained_at': self.get_training_timestamp(),
            'n_users': self.metadata.get('n_users') if self.metadata else 0,
            'n_items': self.metadata.get('n_items') if self.metadata else 0,
            'has_als': self.als_model is not None,
            'has_item_cf': self.item_cf_model is not None,
            'has_popularity': self.popularity_model is not None,
        }


# Global model loader instance (singleton pattern)
_model_loader: Optional[ModelLoader] = None


def get_model_loader(models_path: str = "models") -> ModelLoader:
    """
    Get or create the global model loader instance.
    Uses singleton pattern to ensure models are loaded only once.
    """
    global _model_loader

    if _model_loader is None:
        _model_loader = ModelLoader(models_path)
        _model_loader.load_all()

    return _model_loader


def reset_model_loader():
    """Reset the global model loader (for testing or reloading)."""
    global _model_loader
    _model_loader = None
