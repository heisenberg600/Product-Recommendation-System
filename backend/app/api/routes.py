"""API routes for the recommendation system"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.schemas.recommendation import (
    HealthResponse,
    RecommendationResponse,
    SimilarItemsResponse,
    UserRecommendationRequest,
)
from app.services.recommendation_service import recommendation_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the recommendation service.
    """
    model_version = None
    if recommendation_service.is_initialized and recommendation_service._model_loader:
        model_version = recommendation_service._model_loader.get_model_version()

    return HealthResponse(
        status="healthy" if recommendation_service.is_initialized else "initializing",
        version="1.0.0",
        models_loaded=recommendation_service.is_initialized,
        model_version=model_version
    )


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get product recommendations for a user"
)
async def get_recommendations(
    user_id: str,
    n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return"
    ),
    model: Optional[str] = Query(
        default=None,
        description="Model to use: 'als', 'item_cf', or 'hybrid' (default)"
    ),
    timestamp: Optional[datetime] = Query(
        default=None,
        description="Current timestamp for repurchase cycle calculations (ISO format)"
    ),
    exclude_purchased: bool = Query(
        default=True,
        description="Exclude items the user has already purchased"
    ),
    price_min: Optional[float] = Query(
        default=None,
        ge=0,
        description="Minimum price filter"
    ),
    price_max: Optional[float] = Query(
        default=None,
        ge=0,
        description="Maximum price filter"
    )
) -> RecommendationResponse:
    """
    Get personalized product recommendations for a user.

    ## Model Selection:
    - **als**: ALS Matrix Factorization only
    - **item_cf**: Item-Item Collaborative Filtering only
    - **hybrid** (default): Weighted blend of ALS (40%) + Item-CF (40%) + Popularity (20%)

    ## Two-Level Architecture:
    - **Level 1 (Candidate Generation)**: Generates 200 candidates from selected model(s)
    - **Level 2 (Re-ranking)**: Re-ranks based on:
      - User spending segment
      - Upsell boost (if enabled)
      - Repurchase cycle exclusion
      - Price match scoring

    ## User Types:
    - **Existing users**: Full personalization with two-level architecture
    - **New users**: Popularity-based recommendations filtered by price

    ## Response includes:
    - User information with spending segment
    - List of recommended products with relevance_score and confidence_score
    - Model source and processing time

    ## Examples:
    ```
    GET /api/v1/recommendations/41786230378?n=5&model=als
    GET /api/v1/recommendations/41786230378?n=5&model=item_cf
    GET /api/v1/recommendations/41786230378?n=5&model=hybrid&timestamp=2025-01-15T10:00:00
    ```
    """
    # Validate service is ready
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    # Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=400,
            detail="user_id cannot be empty"
        )

    # Validate price range
    if price_min is not None and price_max is not None and price_min > price_max:
        raise HTTPException(
            status_code=400,
            detail="price_min cannot be greater than price_max"
        )

    try:
        response = recommendation_service.get_recommendations(
            user_id=user_id.strip(),
            num_recommendations=n,
            timestamp=timestamp,
            model_type=model,
            exclude_purchased=exclude_purchased,
            price_range_min=price_min,
            price_range_max=price_max
        )
        return response

    except ValueError as e:
        logger.warning(f"Validation error for user {user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating recommendations for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get recommendations with advanced options"
)
async def get_recommendations_post(
    request: UserRecommendationRequest
) -> RecommendationResponse:
    """
    Get recommendations with advanced filtering options via POST.

    Use this endpoint when you need more control over recommendation parameters,
    including model selection and timestamp for repurchase cycle calculations.

    ## Request Body:
    - user_id: User identifier
    - num_recommendations: Number of recommendations (1-20)
    - model_type: 'als', 'item_cf', or 'hybrid' (default)
    - timestamp: Current timestamp for repurchase cycle (optional)
    - exclude_purchased: Whether to exclude purchased items
    - price_range_min/max: Price filters
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    # Validate price range
    if (request.price_range_min is not None and
        request.price_range_max is not None and
        request.price_range_min > request.price_range_max):
        raise HTTPException(
            status_code=400,
            detail="price_range_min cannot be greater than price_range_max"
        )

    try:
        response = recommendation_service.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            timestamp=request.timestamp,
            model_type=request.model_type,
            exclude_purchased=request.exclude_purchased,
            price_range_min=request.price_range_min,
            price_range_max=request.price_range_max
        )
        return response

    except ValueError as e:
        logger.warning(f"Validation error for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.get(
    "/recommendations/anonymous",
    tags=["Recommendations"],
    summary="Get recommendations for a new visitor"
)
async def get_anonymous_recommendations(
    n: int = Query(default=5, ge=1, le=20, description="Number of recommendations"),
    price_max: Optional[float] = Query(default=None, ge=0, description="Maximum price filter")
) -> dict:
    """
    Get recommendations for a completely new visitor with no purchase history.

    Returns popular items, optionally filtered by price.
    Perfect for first-time visitors or anonymous users.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    # Get popular items filtered by price
    popular_items = recommendation_service._model_loader.get_popular_items(
        n=n * 2,  # Get more to filter
        price_max=price_max
    )

    recommendations = []
    for item_id, score, confidence in popular_items[:n]:
        item_features = recommendation_service._model_loader.get_item_features(item_id)
        item_price = item_features.get('avg_price', 0) if item_features else 0

        recommendations.append({
            "item_id": item_id,
            "relevance_score": round(score, 4),
            "confidence_score": round(confidence, 4),
            "item_price": round(item_price, 2),
            "recommendation_reason": "Popular item - trending among customers",
            "model_source": "popularity"
        })

    return {
        "user_type": "anonymous",
        "recommendations": recommendations,
        "primary_model": "popularity",
        "message": "Recommendations for new visitors based on popular items"
    }


@router.post(
    "/recommendations/custom",
    tags=["Recommendations"],
    summary="Get recommendations based on custom purchase history"
)
async def get_custom_recommendations(
    items: list[str] = [],
    timestamp: Optional[datetime] = None,
    n: int = Query(default=5, ge=1, le=20, description="Number of recommendations")
) -> dict:
    """
    Get recommendations based on a custom list of purchased items.

    Useful for:
    - "What if" scenarios
    - Testing recommendations with different purchase histories
    - Building custom recommendation widgets

    Pass a list of item IDs that represent a simulated purchase history.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    if not items:
        # No items provided, return popular items
        return await get_anonymous_recommendations(n=n)

    # Use Item-CF to find similar items based on the provided items
    item_scores = {}
    model_loader = recommendation_service._model_loader

    for purchased_item in items:
        similar = model_loader.get_similar_items(purchased_item, n=20)
        for similar_item_id, similarity, confidence in similar:
            if similar_item_id not in items:  # Exclude already "purchased" items
                if similar_item_id not in item_scores:
                    item_scores[similar_item_id] = []
                item_scores[similar_item_id].append((similarity, confidence))

    # Aggregate scores
    recommendations = []
    for item_id, scores_list in item_scores.items():
        # Normalize scores
        max_sim = max(s[0] for s in scores_list) if scores_list else 1
        if max_sim > 1:
            scores_list = [(s / max_sim, c) for s, c in scores_list]

        avg_score = sum(s[0] for s in scores_list) / len(scores_list)
        avg_confidence = sum(s[1] for s in scores_list) / len(scores_list)

        item_features = model_loader.get_item_features(item_id)
        item_price = item_features.get('avg_price', 0) if item_features else 0

        recommendations.append({
            "item_id": item_id,
            "relevance_score": round(min(avg_score, 1.0), 4),
            "confidence_score": round(min(avg_confidence, 1.0), 4),
            "item_price": round(item_price, 2),
            "recommendation_reason": f"Similar to {len(scores_list)} items in your selection",
            "model_source": "item_cf"
        })

    # Sort by relevance and take top N
    recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
    recommendations = recommendations[:n]

    # If not enough recommendations, pad with popular items
    if len(recommendations) < n:
        popular = model_loader.get_popular_items(n=n - len(recommendations), exclude_items=items)
        for item_id, score, confidence in popular:
            item_features = model_loader.get_item_features(item_id)
            item_price = item_features.get('avg_price', 0) if item_features else 0
            recommendations.append({
                "item_id": item_id,
                "relevance_score": round(score, 4),
                "confidence_score": round(confidence, 4),
                "item_price": round(item_price, 2),
                "recommendation_reason": "Popular item",
                "model_source": "popularity"
            })

    return {
        "input_items": items,
        "timestamp": timestamp.isoformat() if timestamp else None,
        "recommendations": recommendations,
        "primary_model": "item_cf" if item_scores else "popularity",
        "message": f"Recommendations based on {len(items)} selected items"
    }


@router.get(
    "/items/{item_id}/similar",
    response_model=SimilarItemsResponse,
    tags=["Items"],
    summary="Get similar items"
)
async def get_similar_items(
    item_id: str,
    n: int = Query(default=5, ge=1, le=20, description="Number of similar items")
) -> SimilarItemsResponse:
    """
    Get items similar to a given item based on co-purchase patterns.

    Useful for "Customers who bought this also bought" recommendations.

    ## Response includes:
    - item_id: The source item
    - similar_items: List of similar items with:
      - relevance_score: How similar the item is (0-1)
      - confidence_score: Confidence in the similarity (0-1)
      - item_price: Average price of the item

    ## Example:
    ```
    GET /api/v1/items/7003858505/similar?n=5
    ```
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    response = recommendation_service.get_similar_items(item_id, n)

    if not response.similar_items:
        raise HTTPException(
            status_code=404,
            detail=f"No similar items found for item {item_id}"
        )

    return response


@router.get(
    "/items/popular",
    tags=["Items"],
    summary="Get popular items"
)
async def get_popular_items(
    n: int = Query(default=10, ge=1, le=100, description="Number of items")
) -> dict:
    """
    Get the most popular items based on purchase frequency.

    Useful for homepage recommendations or cold-start scenarios.

    ## Response includes:
    - count: Number of items returned
    - items: List of popular items with:
      - item_id: Product identifier
      - item_price: Average price
      - popularity_score: Normalized popularity (0-1)
      - confidence_score: Confidence in the ranking
      - purchase_count: Number of purchases
      - unique_buyers: Number of unique buyers
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    popular_items = recommendation_service.get_popular_items(n)

    return {
        "count": len(popular_items),
        "items": popular_items
    }


@router.get(
    "/users",
    tags=["Users"],
    summary="Get all users"
)
async def get_users() -> dict:
    """
    Get list of all users categorized by type (loyal vs new).

    Returns user IDs grouped by loyalty status based on original data source.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    if not recommendation_service._model_loader or not recommendation_service._model_loader.user_profiles:
        raise HTTPException(
            status_code=503,
            detail="User profiles not loaded"
        )

    user_profiles = recommendation_service._model_loader.user_profiles

    loyal_users = []
    new_users = []

    for user_id, profile in user_profiles.items():
        # Use is_loyal field from profile (based on original data source)
        if profile.get('is_loyal', False):
            loyal_users.append(user_id)
        else:
            new_users.append(user_id)

    return {
        "loyal_count": len(loyal_users),
        "new_count": len(new_users),
        "users": {
            "loyal": loyal_users[:100],  # Limit to first 100 for performance
            "new": new_users[:100]
        }
    }


@router.get(
    "/users/{user_id}/profile",
    tags=["Users"],
    summary="Get user profile and segment"
)
async def get_user_profile(
    user_id: str
) -> dict:
    """
    Get profile information for a specific user.

    Includes spending segment, purchase history stats, and preferences.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    user_info = recommendation_service._get_user_info(user_id)
    user_profile = None
    if recommendation_service._model_loader:
        user_profile = recommendation_service._model_loader.get_user_profile(user_id)

    return {
        "user_id": user_id,
        "user_info": user_info.model_dump(),
        "profile": user_profile,
        "is_known_user": user_profile is not None
    }


@router.get(
    "/stats",
    tags=["System"],
    summary="Get system statistics"
)
async def get_statistics() -> dict:
    """
    Get comprehensive statistics about the recommendation system.

    Includes model stats, configuration, and cache metrics.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    return recommendation_service.get_statistics()


@router.get(
    "/config",
    tags=["System"],
    summary="Get tuning configuration"
)
async def get_tuning_config() -> dict:
    """
    Get the current tuning configuration for the recommendation system.

    Shows all tunable knobs including:
    - Model weights (ALS, Item-CF, Popularity)
    - Upsell configuration
    - User segment thresholds
    - Repurchase cycle settings
    - Scoring weights
    """
    from app.core.tuning_config import TUNING_CONFIG

    return {
        "config": TUNING_CONFIG,
        "description": {
            "candidate_pool_size": "Number of candidates from Level 1",
            "model_weights": "Weights for ALS, Item-CF, and Popularity models",
            "upsell": "Configuration for upselling higher-priced items",
            "user_segments": "Percentile thresholds for spending segments",
            "repurchase_cycle": "Settings for excluding recently purchased items",
            "scoring_weights": "Weights for Level 2 re-ranking components"
        }
    }


@router.get(
    "/models",
    tags=["System"],
    summary="Get available models"
)
async def get_available_models() -> dict:
    """
    Get list of available recommendation models and their descriptions.
    """
    return {
        "models": [
            {
                "type": "hybrid",
                "name": "Hybrid Two-Level Recommender",
                "description": "Combines ALS, Item-CF, and Popularity with re-ranking. Default choice.",
                "best_for": "All users",
                "weights": {"als": 0.4, "item_cf": 0.4, "popularity": 0.2}
            },
            {
                "type": "als",
                "name": "ALS Matrix Factorization",
                "description": "Learns latent factors from implicit feedback (spending/quantity ratios).",
                "best_for": "Existing users with purchase history"
            },
            {
                "type": "item_cf",
                "name": "Item-Item Collaborative Filtering",
                "description": "Recommends items similar to user's purchase history using co-purchase patterns.",
                "best_for": "Users with diverse purchase history"
            },
            {
                "type": "popularity",
                "name": "Popularity-based",
                "description": "Recommends popular items weighted by purchases, unique buyers, and quantity.",
                "best_for": "New users (cold-start)"
            }
        ],
        "architecture": {
            "level_1": "Candidate Generation - Fetches 200 candidates from model blend",
            "level_2": "Re-ranking - Applies user segment, upsell boost, repurchase cycle exclusion"
        }
    }
