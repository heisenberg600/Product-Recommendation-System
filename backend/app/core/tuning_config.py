"""
Tuning Configuration for Recommendation System

All tunable knobs and scaling factors are centralized here.
Modify these values to tune the recommendation behavior without code changes.
"""

TUNING_CONFIG = {
    # ==========================================================================
    # CANDIDATE GENERATION
    # ==========================================================================
    "candidate_pool_size": 200,  # Number of candidates from Level 1

    # ==========================================================================
    # MODEL WEIGHTS (for existing users - weighted blend)
    # ==========================================================================
    "model_weights": {
        "als": 0.4,           # ALS Matrix Factorization weight
        "item_cf": 0.4,       # Item-Item Collaborative Filtering weight
        "popularity": 0.2     # Popularity model weight
    },

    # ==========================================================================
    # UPSELL CONFIGURATION
    # Enable to boost higher-priced items for revenue optimization
    # ==========================================================================
    "upsell": {
        "enabled": False,
        "factor": 0.1,              # 0 = no upsell, 1 = max upsell
        "price_boost_weight": 0.2,  # How much to weight price in final score
        "max_price_ratio": 2.0      # Don't recommend items > 2x user's avg
    },

    # ==========================================================================
    # USER SEGMENT THRESHOLDS (percentiles of avg item price)
    # ==========================================================================
    "user_segments": {
        "small": 0.25,        # Bottom 25% - lowest spenders
        "low_average": 0.50,  # 25-50% - below median
        "average": 0.75,      # 50-75% - above median
        "high": 1.0           # Top 25% - highest spenders
    },

    # ==========================================================================
    # REPURCHASE CYCLE CONFIGURATION
    # Boost items when time since purchase > avg repurchase cycle
    # ==========================================================================
    "repurchase_cycle": {
        "enabled": True,
        "default_cycle_days": 30,       # Default if no cycle data
        "min_purchases_for_cycle": 2,   # Min purchases to compute cycle
        "exclusion_buffer_factor": 0.5, # Exclude if days_since < avg_cycle * factor (too recent)
        "boost_enabled": True,          # Enable boosting items due for repurchase
        "boost_factor": 0.15,           # Score boost when item is due (0.15 = +15%)
        "boost_max_ratio": 2.0          # Max boost when days_since = 2x avg_cycle
    },

    # ==========================================================================
    # SCORING WEIGHTS FOR LEVEL 2 RE-RANKING
    # Relevance score = model_score adjusted by these factors
    # ==========================================================================
    "scoring_weights": {
        "model_score": 0.60,      # Primary weight for model's raw score
        "popularity_boost": 0.15, # Boost for popular items (multiply, not add)
        "recency_boost": 0.10,    # Boost for recently purchased items (by others)
        "price_match_boost": 0.15 # Boost for items matching user's price preference
    },

    # ==========================================================================
    # PRICE SENSITIVITY CONFIGURATION
    # Applies soft penalty to items beyond user's price range (not hard filtering)
    # ==========================================================================
    "price_sensitivity": {
        "enabled": True,                     # Enable price sensitivity scoring
        "budget_tolerance": 3,               # No penalty up to 1.5x user's avg price
        "penalty_start_ratio": 1.5,          # Start penalizing above this ratio
        "max_penalty_ratio": 3.0,            # Maximum ratio to consider (above = full penalty)
        "penalty_factor": 0.3,               # Max score reduction (0.3 = reduce by 30%)
        "penalty_curve": "linear",           # "linear" or "exponential"
        "new_user_default_segment": "average",
        "strict_budget_for_new_users": False # Soft penalty for new users too
    },

    # ==========================================================================
    # NEW USER HANDLING
    # ==========================================================================
    "new_user": {
        "strategy": "popularity_filtered_by_price",
        "infer_budget_from_first_purchase": True,
        "default_price_percentile": 50,   # Use median price as default
        "min_interactions_for_personalization": 5
    },

    # ==========================================================================
    # POPULARITY MODEL WEIGHTS
    # How to weight different signals in popularity score
    # ==========================================================================
    "popularity_weights": {
        "purchase_count": 0.4,    # Times purchased (transactions)
        "unique_buyers": 0.4,     # Number of unique buyers
        "total_quantity": 0.2     # Total quantity sold
    },

    # ==========================================================================
    # IMPLICIT FEEDBACK CONFIGURATION
    # How to compute implicit feedback signals
    # ==========================================================================
    "implicit_feedback": {
        "spending_ratio_weight": 0.5,   # Weight for spending ratio signal
        "quantity_ratio_weight": 0.5,   # Weight for quantity ratio signal
        "min_signal": 0.1,              # Minimum implicit signal value
        "max_signal": 5.0,              # Maximum implicit signal value (cap outliers)
        "use_log_transform": True       # Apply log transform to smooth ratios
    },

    # ==========================================================================
    # ITEM-CF CONFIGURATION
    # ==========================================================================
    "item_cf": {
        "similarity_top_k": 100,         # Store top K similar items per item
        "min_co_purchase_count": 2,      # Min co-purchases to consider
        "similarity_threshold": 0.01,    # Min similarity to include
        "use_weighted_similarity": True  # Weight by spending ratios
    },

    # ==========================================================================
    # ALS CONFIGURATION
    # ==========================================================================
    "als": {
        "n_factors": 64,           # Number of latent factors
        "n_iterations": 15,        # Training iterations
        "regularization": 0.01,    # L2 regularization
        "alpha": 40,               # Confidence scaling factor
        "use_gpu": False           # Use GPU if available
    },

    # ==========================================================================
    # CONFIDENCE SCORE CALCULATION
    # Different user types get different confidence levels based on data quality
    # ==========================================================================
    "confidence": {
        "base_confidence": 0.5,           # Base confidence for all recommendations
        "history_factor_weight": 0.3,     # Weight for user history depth
        "model_agreement_weight": 0.2,    # Weight for model agreement
        "min_confidence": 0.1,            # Minimum confidence score
        "max_confidence": 0.99,           # Maximum confidence score
        # User type confidence multipliers (applied to base confidence)
        "user_type_multipliers": {
            "loyal": 1.0,               # Full confidence - user in training data
            "new_with_history": 0.5,    # Reduced - has history but not in training
            "anonymous": 0.3            # Low confidence - no personalization possible
        }
    },

    # ==========================================================================
    # SIMILAR ITEMS ENDPOINT
    # ==========================================================================
    "similar_items": {
        "default_n": 5,           # Default number of similar items
        "max_n": 20,              # Maximum allowed
        "include_price_info": True
    },

    # ==========================================================================
    # CACHING
    # ==========================================================================
    "cache": {
        "enabled": True,
        "ttl_seconds": 3600,          # 1 hour default
        "user_cache_ttl": 1800,       # 30 min for user-specific
        "popular_items_ttl": 7200     # 2 hours for popular items
    },

    # ==========================================================================
    # MODEL VERSIONING
    # ==========================================================================
    "model_version": {
        "current": "1.0.0",
        "require_exact_match": False  # Allow minor version mismatches
    }
}


def get_config(path: str = None):
    """
    Get configuration value by dot-notation path.

    Example:
        get_config("upsell.factor")  # Returns 0.1
        get_config("model_weights.als")  # Returns 0.4
    """
    if path is None:
        return TUNING_CONFIG

    keys = path.split(".")
    value = TUNING_CONFIG
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Config path not found: {path}")
    return value


def update_config(path: str, value):
    """
    Update configuration value by dot-notation path.

    Example:
        update_config("upsell.factor", 0.2)
    """
    keys = path.split(".")
    config = TUNING_CONFIG
    for key in keys[:-1]:
        config = config[key]
    config[keys[-1]] = value
