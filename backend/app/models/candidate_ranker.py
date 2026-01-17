"""
Candidate Ranker for Two-Level Recommendation System

Level 2 re-ranking component that takes candidates from Level 1 and:
1. Filters by user spending segment
2. Applies upsell boost for revenue optimization
3. Excludes items based on repurchase cycle
4. Computes final relevance and confidence scores

Usage:
    from backend.app.models.candidate_ranker import CandidateRanker

    ranker = CandidateRanker(config, model_loader)
    final_recs = ranker.rank(candidates, user_id, timestamp, n=5)
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CandidateRanker:
    """
    Level 2 re-ranker for recommendation candidates.

    Takes candidates from Level 1 (ALS + Item-CF + Popularity blend)
    and applies business logic for final ranking.
    """

    def __init__(self, config: Dict, model_loader: Any):
        """
        Initialize the ranker.

        Args:
            config: Tuning configuration dictionary
            model_loader: ModelLoader instance with loaded models
        """
        self.config = config
        self.model_loader = model_loader

    def rank(
        self,
        candidates: List[Tuple[str, float, float, str]],
        user_id: str,
        timestamp: datetime = None,
        n: int = 5,
        exclude_items: List[str] = None
    ) -> List[Dict]:
        """
        Rank candidates and return final recommendations.

        Args:
            candidates: List of (item_id, base_score, confidence, source) from Level 1
            user_id: User identifier
            timestamp: Current request timestamp (for repurchase cycle)
            n: Number of final recommendations to return
            exclude_items: Items to exclude from recommendations

        Returns:
            List of recommendation dicts with item_id, relevance_score,
            confidence_score, item_price, and recommendation_reason
        """
        if timestamp is None:
            timestamp = datetime.now()

        exclude_items = exclude_items or []

        # Get user information
        user_profile = self.model_loader.get_user_profile(user_id)
        user_segment = self.model_loader.get_user_segment(user_id)
        user_avg_price = self.model_loader.get_user_avg_price(user_id)

        is_new_user = user_profile is None

        if is_new_user:
            # Use default segment for new users
            user_segment = self.config.get('price_sensitivity', {}).get(
                'new_user_default_segment', 'average'
            )
            # Infer budget from segment price range
            price_range = self.model_loader.get_segment_price_range(user_segment)
            user_avg_price = (price_range[0] + min(price_range[1], 100)) / 2

        # Get scoring weights
        scoring_weights = self.config.get('scoring_weights', {})
        base_weight = scoring_weights.get('base_relevance', 0.4)
        price_match_weight = scoring_weights.get('price_match', 0.3)
        popularity_weight = scoring_weights.get('popularity', 0.2)
        recency_weight = scoring_weights.get('recency', 0.1)

        # Get upsell config
        upsell_config = self.config.get('upsell', {})
        upsell_enabled = upsell_config.get('enabled', False)
        upsell_factor = upsell_config.get('factor', 0.1)
        price_boost_weight = upsell_config.get('price_boost_weight', 0.2)
        max_price_ratio = upsell_config.get('max_price_ratio', 2.0)

        # Get repurchase config
        repurchase_config = self.config.get('repurchase_cycle', {})
        repurchase_enabled = repurchase_config.get('enabled', True)
        cycle_buffer_factor = repurchase_config.get('cycle_buffer_factor', 0.8)

        # Budget tolerance
        budget_tolerance = self.config.get('price_sensitivity', {}).get(
            'budget_tolerance', 1.5
        )

        # Score and filter candidates
        scored_candidates = []

        for item_id, base_score, base_confidence, source in candidates:
            # Skip excluded items
            if item_id in exclude_items:
                continue

            # Get item features
            item_features = self.model_loader.get_item_features(item_id)
            if not item_features:
                continue

            item_price = item_features.get('avg_price', 0)
            item_popularity = item_features.get('popularity_score', 0)
            item_recency = item_features.get('recency_score', 0)

            # Check repurchase cycle exclusion
            if repurchase_enabled and not is_new_user:
                if self._should_exclude_repurchase(
                    item_id, user_id, timestamp, cycle_buffer_factor
                ):
                    continue

            # Check price is within budget (with tolerance)
            if user_avg_price > 0:
                max_allowed_price = user_avg_price * budget_tolerance
                if item_price > max_allowed_price:
                    # Skip items way out of budget
                    continue

            # Compute price match score (0-1)
            price_match_score = self._compute_price_match(
                item_price, user_avg_price, user_segment
            )

            # Compute upsell boost
            upsell_boost = 0.0
            if upsell_enabled and user_avg_price > 0:
                upsell_boost = self._compute_upsell_boost(
                    item_price, user_avg_price, upsell_factor,
                    price_boost_weight, max_price_ratio
                )

            # Compute final relevance score
            relevance_score = (
                base_weight * base_score +
                price_match_weight * price_match_score +
                popularity_weight * item_popularity +
                recency_weight * item_recency +
                upsell_boost
            )

            # Normalize to 0-1
            relevance_score = np.clip(relevance_score, 0, 1)

            # Compute final confidence score
            # Combine base confidence with price match certainty
            confidence_score = self._compute_confidence(
                base_confidence, price_match_score, is_new_user
            )

            # Generate recommendation reason
            reason = self._generate_reason(
                source, base_score, price_match_score, upsell_boost > 0
            )

            scored_candidates.append({
                'item_id': item_id,
                'relevance_score': round(relevance_score, 4),
                'confidence_score': round(confidence_score, 4),
                'item_price': round(item_price, 2),
                'recommendation_reason': reason,
                'model_source': source,
                '_base_score': base_score,
                '_price_match': price_match_score
            })

        # Sort by relevance score
        scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return top N
        return scored_candidates[:n]

    def _should_exclude_repurchase(
        self,
        item_id: str,
        user_id: str,
        timestamp: datetime,
        buffer_factor: float
    ) -> bool:
        """
        Check if item was purchased too recently based on repurchase cycle.

        Returns True if item should be excluded.
        """
        # Get user's last purchase of this item
        last_purchase = self.model_loader.get_user_item_last_purchase(user_id, item_id)
        if not last_purchase:
            return False

        # Get item's average repurchase cycle
        avg_cycle = self.model_loader.get_avg_repurchase_days(item_id)

        # Compute days since last purchase
        days_since = (timestamp - last_purchase).days

        # Exclude if purchased too recently
        threshold = avg_cycle * buffer_factor
        if days_since < threshold:
            logger.debug(
                f"Excluding {item_id} for user {user_id}: "
                f"purchased {days_since} days ago, cycle is {avg_cycle}"
            )
            return True

        return False

    def _compute_price_match(
        self,
        item_price: float,
        user_avg_price: float,
        user_segment: str
    ) -> float:
        """
        Compute how well the item price matches user's spending pattern.

        Returns a score from 0 to 1.
        """
        if user_avg_price <= 0:
            return 0.5  # Neutral for unknown price

        # Get segment price range
        min_price, max_price = self.model_loader.get_segment_price_range(user_segment)

        # Check if item is in segment's price range
        if min_price <= item_price <= max_price:
            return 1.0

        # Compute distance from ideal range
        if item_price < min_price:
            # Item is cheaper - still good
            ratio = item_price / min_price if min_price > 0 else 1
            return max(ratio, 0.5)  # Floor at 0.5 for cheaper items
        else:
            # Item is more expensive - penalize more
            ratio = max_price / item_price if item_price > 0 else 0
            return max(ratio, 0.0)

    def _compute_upsell_boost(
        self,
        item_price: float,
        user_avg_price: float,
        upsell_factor: float,
        price_boost_weight: float,
        max_price_ratio: float
    ) -> float:
        """
        Compute upsell score boost for higher-priced items.

        Returns additional score to add (0 to price_boost_weight * upsell_factor).
        """
        if user_avg_price <= 0 or item_price <= user_avg_price:
            return 0.0

        price_ratio = item_price / user_avg_price

        # Don't boost items that are too expensive
        if price_ratio > max_price_ratio:
            return 0.0

        # Linear boost from 1x to max_price_ratio
        # At 1x: boost = 0
        # At max_price_ratio: boost = upsell_factor * price_boost_weight
        normalized_ratio = (price_ratio - 1) / (max_price_ratio - 1)
        boost = normalized_ratio * upsell_factor * price_boost_weight

        return boost

    def _compute_confidence(
        self,
        base_confidence: float,
        price_match_score: float,
        is_new_user: bool
    ) -> float:
        """
        Compute final confidence score.

        Combines model confidence with price match and user status.
        """
        confidence_config = self.config.get('confidence', {})
        base_conf_weight = confidence_config.get('base_confidence', 0.5)
        history_weight = confidence_config.get('history_factor_weight', 0.3)
        min_conf = confidence_config.get('min_confidence', 0.1)
        max_conf = confidence_config.get('max_confidence', 0.99)

        # Start with base confidence
        confidence = base_confidence

        # Boost by price match
        confidence = confidence * 0.7 + price_match_score * 0.3

        # Reduce confidence for new users
        if is_new_user:
            confidence *= 0.7

        # Clamp to range
        confidence = np.clip(confidence, min_conf, max_conf)

        return confidence

    def _generate_reason(
        self,
        source: str,
        base_score: float,
        price_match_score: float,
        has_upsell: bool
    ) -> str:
        """Generate a human-readable recommendation reason."""
        reasons = []

        # Model-based reason
        if source == 'als':
            reasons.append("Matches your preferences")
        elif source == 'item_cf':
            reasons.append("Bought with similar items")
        elif source == 'popularity':
            reasons.append("Popular among customers")
        elif source == 'blend':
            reasons.append("Personalized for you")

        # Price-based reason
        if price_match_score >= 0.8:
            reasons.append("fits your budget")
        elif price_match_score >= 0.5:
            reasons.append("good value")

        # Upsell reason
        if has_upsell:
            reasons.append("premium choice")

        if reasons:
            return reasons[0].capitalize() + (
                " - " + ", ".join(reasons[1:]) if len(reasons) > 1 else ""
            )
        return "Recommended for you"

    def rank_for_new_user(
        self,
        first_purchase_price: float = None,
        n: int = 5,
        exclude_items: List[str] = None
    ) -> List[Dict]:
        """
        Get recommendations for a new user based on popularity and inferred budget.

        Args:
            first_purchase_price: Price of user's first purchase (if available)
            n: Number of recommendations
            exclude_items: Items to exclude

        Returns:
            List of recommendation dicts
        """
        exclude_items = exclude_items or []

        # Infer price range from first purchase or use default
        new_user_config = self.config.get('new_user', {})
        infer_budget = new_user_config.get('infer_budget_from_first_purchase', True)
        default_percentile = new_user_config.get('default_price_percentile', 50)
        budget_tolerance = self.config.get('price_sensitivity', {}).get(
            'budget_tolerance', 1.5
        )

        if first_purchase_price and infer_budget:
            # Infer budget from first purchase
            price_min = first_purchase_price * 0.5
            price_max = first_purchase_price * budget_tolerance
        else:
            # Use average segment as default
            price_min, price_max = self.model_loader.get_segment_price_range('average')

        # Get popular items within price range
        popular_items = self.model_loader.get_popular_items(
            n=n * 2,  # Get extra to account for filtering
            exclude_items=exclude_items,
            price_min=price_min,
            price_max=price_max
        )

        # Format results
        results = []
        for item_id, popularity_score, confidence in popular_items[:n]:
            item_features = self.model_loader.get_item_features(item_id)
            item_price = item_features.get('avg_price', 0) if item_features else 0

            results.append({
                'item_id': item_id,
                'relevance_score': round(popularity_score, 4),
                'confidence_score': round(confidence * 0.7, 4),  # Lower confidence for new users
                'item_price': round(item_price, 2),
                'recommendation_reason': "Popular item in your price range",
                'model_source': 'popularity'
            })

        return results
