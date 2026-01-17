"""
Candidate Ranker for Two-Level Recommendation System

Level 2 re-ranking component that takes candidates from Level 1 and:
1. Applies soft price sensitivity penalty (not hard filtering)
2. Boosts items due for repurchase based on product-specific cycles
3. Applies upsell boost for revenue optimization
4. Computes final relevance and confidence scores

Relevance Score Calculation:
- Base relevance from model (ALS, Item-CF, or Hybrid blend)
- Adjusted by price sensitivity penalty (for expensive items)
- Boosted by repurchase cycle (if item is due for repurchase)
- Optional upsell boost for premium items

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

        Relevance Score Calculation:
        1. Start with model's base score (ALS, Item-CF, or Hybrid)
        2. Apply soft price penalty if item is expensive (not hard filter)
        3. Apply repurchase cycle boost if item is due for repurchase
        4. Apply optional upsell boost for premium items
        5. Apply small popularity and recency boosts

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
        model_score_weight = scoring_weights.get('model_score', 0.60)
        popularity_boost_weight = scoring_weights.get('popularity_boost', 0.15)
        recency_boost_weight = scoring_weights.get('recency_boost', 0.10)
        price_match_boost_weight = scoring_weights.get('price_match_boost', 0.15)

        # Get price sensitivity config (for soft penalty)
        price_config = self.config.get('price_sensitivity', {})
        price_sensitivity_enabled = price_config.get('enabled', True)
        penalty_start_ratio = price_config.get('penalty_start_ratio', 1.5)
        max_penalty_ratio = price_config.get('max_penalty_ratio', 3.0)
        penalty_factor = price_config.get('penalty_factor', 0.3)

        # Get upsell config
        upsell_config = self.config.get('upsell', {})
        upsell_enabled = upsell_config.get('enabled', False)
        upsell_factor = upsell_config.get('factor', 0.1)
        price_boost_weight = upsell_config.get('price_boost_weight', 0.2)
        max_price_ratio = upsell_config.get('max_price_ratio', 2.0)

        # Get repurchase config
        repurchase_config = self.config.get('repurchase_cycle', {})
        repurchase_enabled = repurchase_config.get('enabled', True)
        exclusion_buffer_factor = repurchase_config.get('exclusion_buffer_factor', 0.5)
        boost_enabled = repurchase_config.get('boost_enabled', True)
        boost_factor = repurchase_config.get('boost_factor', 0.15)
        boost_max_ratio = repurchase_config.get('boost_max_ratio', 2.0)

        # Score candidates
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

            # Track adjustments for reason generation
            price_penalty_applied = False
            repurchase_boost_applied = False
            upsell_boost_applied = False

            # =================================================================
            # STEP 1: Start with model's base relevance score
            # =================================================================
            # Base score is already 0-1 from model (ALS sigmoid, Item-CF normalized)
            relevance_score = base_score * model_score_weight

            # =================================================================
            # STEP 2: Apply soft price sensitivity penalty
            # (Downgrade expensive items instead of filtering)
            # =================================================================
            price_penalty = 0.0
            if price_sensitivity_enabled and user_avg_price > 0:
                price_penalty = self._compute_price_penalty(
                    item_price, user_avg_price,
                    penalty_start_ratio, max_penalty_ratio, penalty_factor
                )
                if price_penalty > 0:
                    price_penalty_applied = True

            # =================================================================
            # STEP 3: Apply repurchase cycle logic
            # - Exclude if purchased too recently (< exclusion_buffer_factor * avg_cycle)
            # - BOOST if item is due for repurchase (> avg_cycle)
            # =================================================================
            repurchase_boost = 0.0
            if repurchase_enabled and not is_new_user:
                # Check if should exclude (too recent)
                if self._should_exclude_repurchase(
                    item_id, user_id, timestamp, exclusion_buffer_factor
                ):
                    continue

                # Check if should boost (due for repurchase)
                if boost_enabled:
                    repurchase_boost = self._compute_repurchase_boost(
                        item_id, user_id, timestamp,
                        boost_factor, boost_max_ratio
                    )
                    if repurchase_boost > 0:
                        repurchase_boost_applied = True

            # =================================================================
            # STEP 4: Apply optional upsell boost
            # =================================================================
            upsell_boost = 0.0
            if upsell_enabled and user_avg_price > 0:
                upsell_boost = self._compute_upsell_boost(
                    item_price, user_avg_price, upsell_factor,
                    price_boost_weight, max_price_ratio
                )
                if upsell_boost > 0:
                    upsell_boost_applied = True

            # =================================================================
            # STEP 5: Apply popularity and recency boosts
            # =================================================================
            popularity_boost = item_popularity * popularity_boost_weight
            recency_boost = item_recency * recency_boost_weight

            # Compute price match score for additional boost
            price_match_score = self._compute_price_match(
                item_price, user_avg_price, user_segment
            )
            price_match_boost = price_match_score * price_match_boost_weight

            # =================================================================
            # FINAL: Combine all factors
            # =================================================================
            relevance_score = (
                relevance_score +
                popularity_boost +
                recency_boost +
                price_match_boost +
                repurchase_boost +
                upsell_boost -
                price_penalty  # Subtract penalty
            )

            # Normalize to 0-1
            relevance_score = np.clip(relevance_score, 0, 1)

            # Compute final confidence score
            confidence_score = self._compute_confidence(
                base_confidence, price_match_score, is_new_user
            )

            # Generate recommendation reason
            reason = self._generate_reason(
                source=source,
                base_score=base_score,
                price_match_score=price_match_score,
                has_upsell=upsell_boost_applied,
                has_repurchase_boost=repurchase_boost_applied,
                has_price_penalty=price_penalty_applied
            )

            scored_candidates.append({
                'item_id': item_id,
                'relevance_score': round(relevance_score, 4),
                'confidence_score': round(confidence_score, 4),
                'item_price': round(item_price, 2),
                'recommendation_reason': reason,
                'model_source': source,
                '_base_score': base_score,
                '_price_match': price_match_score,
                '_price_penalty': round(price_penalty, 4),
                '_repurchase_boost': round(repurchase_boost, 4)
            })

        # Sort by relevance score
        scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Return top N
        return scored_candidates[:n]

    def _compute_price_penalty(
        self,
        item_price: float,
        user_avg_price: float,
        penalty_start_ratio: float,
        max_penalty_ratio: float,
        penalty_factor: float
    ) -> float:
        """
        Compute soft price penalty for expensive items.

        Instead of hard filtering, we reduce the score of items that are
        beyond the user's typical price range.

        Args:
            item_price: Price of the item
            user_avg_price: User's average item price
            penalty_start_ratio: Start penalizing above this ratio (e.g., 1.5x)
            max_penalty_ratio: Full penalty at this ratio (e.g., 3x)
            penalty_factor: Maximum penalty to apply (e.g., 0.3 = 30% reduction)

        Returns:
            Penalty value to subtract from relevance score (0 to penalty_factor)
        """
        if user_avg_price <= 0 or item_price <= 0:
            return 0.0

        price_ratio = item_price / user_avg_price

        # No penalty if within acceptable range
        if price_ratio <= penalty_start_ratio:
            return 0.0

        # Calculate penalty (linear interpolation)
        # At penalty_start_ratio: penalty = 0
        # At max_penalty_ratio: penalty = penalty_factor
        penalty_range = max_penalty_ratio - penalty_start_ratio
        if penalty_range <= 0:
            return 0.0

        # How far into the penalty zone
        penalty_progress = (price_ratio - penalty_start_ratio) / penalty_range
        penalty_progress = min(penalty_progress, 1.0)  # Cap at 100%

        penalty = penalty_progress * penalty_factor

        logger.debug(
            f"Price penalty for item at ${item_price:.2f} "
            f"(user avg: ${user_avg_price:.2f}, ratio: {price_ratio:.2f}): {penalty:.4f}"
        )

        return penalty

    def _compute_repurchase_boost(
        self,
        item_id: str,
        user_id: str,
        timestamp: datetime,
        boost_factor: float,
        boost_max_ratio: float
    ) -> float:
        """
        Compute boost for items that are due for repurchase.

        If the user bought an item in the past and the time since purchase
        exceeds the item's average repurchase cycle, boost its score.

        Args:
            item_id: Item identifier
            user_id: User identifier
            timestamp: Current timestamp
            boost_factor: Maximum boost to apply (e.g., 0.15 = 15% boost)
            boost_max_ratio: Full boost when days_since = boost_max_ratio * avg_cycle

        Returns:
            Boost value to add to relevance score (0 to boost_factor)
        """
        # Get user's last purchase of this item
        last_purchase = self.model_loader.get_user_item_last_purchase(user_id, item_id)
        if not last_purchase:
            return 0.0

        # Get item's average repurchase cycle
        avg_cycle = self.model_loader.get_avg_repurchase_days(item_id)
        if avg_cycle <= 0:
            return 0.0

        # Compute days since last purchase
        days_since = (timestamp - last_purchase).days

        # Only boost if past the repurchase cycle
        if days_since <= avg_cycle:
            return 0.0

        # Calculate boost (linear interpolation)
        # At days_since = avg_cycle: boost = 0
        # At days_since = avg_cycle * boost_max_ratio: boost = boost_factor
        max_days = avg_cycle * boost_max_ratio
        boost_range = max_days - avg_cycle

        if boost_range <= 0:
            return 0.0

        # How far past the cycle
        boost_progress = (days_since - avg_cycle) / boost_range
        boost_progress = min(boost_progress, 1.0)  # Cap at 100%

        boost = boost_progress * boost_factor

        logger.debug(
            f"Repurchase boost for item {item_id}: "
            f"days_since={days_since}, avg_cycle={avg_cycle:.1f}, boost={boost:.4f}"
        )

        return boost

    def _should_exclude_repurchase(
        self,
        item_id: str,
        user_id: str,
        timestamp: datetime,
        exclusion_buffer_factor: float
    ) -> bool:
        """
        Check if item was purchased too recently based on repurchase cycle.

        Exclusion logic: If days_since_purchase < avg_cycle * exclusion_buffer_factor,
        the item was bought too recently and should be excluded.

        Example: If avg_cycle=30 days and exclusion_buffer_factor=0.5,
        exclude if purchased within the last 15 days.

        Returns True if item should be excluded.
        """
        # Get user's last purchase of this item
        last_purchase = self.model_loader.get_user_item_last_purchase(user_id, item_id)
        if not last_purchase:
            return False

        # Get item's average repurchase cycle
        avg_cycle = self.model_loader.get_avg_repurchase_days(item_id)
        if avg_cycle <= 0:
            return False

        # Compute days since last purchase
        days_since = (timestamp - last_purchase).days

        # Exclude if purchased too recently
        exclusion_threshold = avg_cycle * exclusion_buffer_factor
        if days_since < exclusion_threshold:
            logger.debug(
                f"Excluding {item_id} for user {user_id}: "
                f"purchased {days_since} days ago, exclusion threshold is {exclusion_threshold:.1f} days "
                f"(avg cycle: {avg_cycle:.1f} days)"
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
        has_upsell: bool = False,
        has_repurchase_boost: bool = False,
        has_price_penalty: bool = False
    ) -> str:
        """Generate a human-readable recommendation reason."""
        reasons = []

        # Primary reason based on why item was recommended
        if has_repurchase_boost:
            reasons.append("Time to restock")
        elif source == 'als':
            reasons.append("Matches your preferences")
        elif source == 'item_cf':
            reasons.append("Bought with similar items")
        elif source == 'popularity':
            reasons.append("Popular among customers")
        elif source == 'hybrid':
            reasons.append("Personalized for you")
        else:
            reasons.append("Recommended for you")

        # Price-based reason
        if price_match_score >= 0.8:
            reasons.append("fits your budget")
        elif price_match_score >= 0.5:
            reasons.append("good value")
        elif has_price_penalty:
            reasons.append("premium item")

        # Upsell reason
        if has_upsell and not has_price_penalty:
            reasons.append("upgrade choice")

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
