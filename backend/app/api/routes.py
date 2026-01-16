"""API routes for the recommendation system"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.schemas.recommendation import (
    HealthResponse,
    ModelType,
    RecommendationResponse,
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
    return HealthResponse(
        status="healthy" if recommendation_service.is_initialized else "initializing",
        version="1.0.0",
        models_loaded=recommendation_service.is_initialized,
        data_loaded=recommendation_service.data_loaded
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
    model: Optional[ModelType] = Query(
        default=None,
        description="Specific model to use (item_cf, matrix_factorization, popularity, price_segment, hybrid)"
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

    The system automatically determines the best recommendation strategy based on:
    - **Loyal customers**: Uses collaborative filtering and matrix factorization for personalization
    - **New customers**: Uses popularity and price-based recommendations for cold-start

    ## Response includes:
    - User information and classification
    - List of recommended products with relevance and confidence scores
    - Model used and processing time

    ## Example:
    ```
    GET /api/v1/recommendations/41786230378?n=5
    ```
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    try:
        response = recommendation_service.get_recommendations(
            user_id=user_id,
            num_recommendations=n,
            model_type=model,
            exclude_purchased=exclude_purchased,
            price_range_min=price_min,
            price_range_max=price_max
        )
        return response

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
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

    Use this endpoint when you need more control over recommendation parameters.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    try:
        response = recommendation_service.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            model_type=request.model_type,
            exclude_purchased=request.exclude_purchased,
            price_range_min=request.price_range_min,
            price_range_max=request.price_range_max
        )
        return response

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.get(
    "/items/{item_id}/similar",
    tags=["Items"],
    summary="Get similar items"
)
async def get_similar_items(
    item_id: str,
    n: int = Query(default=10, ge=1, le=50, description="Number of similar items")
) -> dict:
    """
    Get items similar to a given item based on co-purchase patterns.

    Useful for "Customers who bought this also bought" recommendations.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    similar_items = recommendation_service.get_similar_items(item_id, n)

    if not similar_items:
        raise HTTPException(
            status_code=404,
            detail=f"No similar items found for item {item_id}"
        )

    return {
        "item_id": item_id,
        "similar_items": similar_items
    }


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
    summary="Get all user IDs"
)
async def get_all_users() -> dict:
    """
    Get all user IDs grouped by type (loyal/new).

    Useful for testing and validation.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    users = recommendation_service.get_all_users()

    return {
        "loyal_count": len(users["loyal"]),
        "new_count": len(users["new"]),
        "users": users
    }


@router.get(
    "/users/{user_id}/history",
    tags=["Users"],
    summary="Get user purchase history"
)
async def get_user_history(
    user_id: str,
    limit: int = Query(default=50, ge=1, le=500, description="Maximum items to return")
) -> dict:
    """
    Get purchase history for a specific user.

    Useful for understanding user preferences and validating recommendations.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    from app.services.recommendation_service import recommendation_service

    user_info = recommendation_service._get_user_info(user_id)
    history = recommendation_service._data_loader.get_user_history(user_id)

    if history.empty:
        return {
            "user_id": user_id,
            "user_info": user_info.model_dump(),
            "history": [],
            "total_count": 0
        }

    # Convert to list of dicts
    history_list = history.head(limit).to_dict(orient="records")

    # Clean up datetime for JSON serialization
    for item in history_list:
        if "ticket_datetime" in item:
            item["ticket_datetime"] = str(item["ticket_datetime"])

    return {
        "user_id": user_id,
        "user_info": user_info.model_dump(),
        "history": history_list,
        "total_count": len(history)
    }


@router.get(
    "/stats",
    tags=["System"],
    summary="Get system statistics"
)
async def get_statistics() -> dict:
    """
    Get comprehensive statistics about the recommendation system.

    Includes data statistics, model performance, and cache metrics.
    """
    if not recommendation_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again shortly."
        )

    return recommendation_service.get_statistics()


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
                "name": "Hybrid Recommender",
                "description": "Combines multiple models for best results. Default choice.",
                "best_for": "All users"
            },
            {
                "type": "item_cf",
                "name": "Item-Item Collaborative Filtering",
                "description": "Recommends items similar to user's purchase history.",
                "best_for": "Loyal customers with purchase history"
            },
            {
                "type": "matrix_factorization",
                "name": "Matrix Factorization (ALS)",
                "description": "Learns latent factors from user-item interactions.",
                "best_for": "Loyal customers with diverse history"
            },
            {
                "type": "popularity",
                "name": "Popularity-based",
                "description": "Recommends trending and popular items.",
                "best_for": "New customers (cold-start)"
            },
            {
                "type": "price_segment",
                "name": "Price Segment",
                "description": "Matches recommendations to user's price preferences.",
                "best_for": "Price-sensitive recommendations"
            }
        ]
    }
