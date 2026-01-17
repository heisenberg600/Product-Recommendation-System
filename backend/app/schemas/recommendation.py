"""Pydantic schemas for recommendation API"""

from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator


class UserType(str, Enum):
    """User segment type"""
    LOYAL = "loyal"
    NEW = "new"
    UNKNOWN = "unknown"


class SpendingSegment(str, Enum):
    """User spending segment (based on average item price percentiles)"""
    SMALL = "small"           # Bottom 25% - lowest spenders
    LOW_AVERAGE = "low_average"  # 25-50% - below median
    AVERAGE = "average"       # 50-75% - above median
    HIGH = "high"             # Top 25% - highest spenders


class ModelType(str, Enum):
    """Recommendation model type"""
    ITEM_CF = "item_cf"
    MATRIX_FACTORIZATION = "matrix_factorization"
    ALS = "als"
    POPULARITY = "popularity"
    HYBRID = "hybrid"
    PRICE_SEGMENT = "price_segment"
    BLEND = "blend"


class RecommendationItem(BaseModel):
    """Individual product recommendation"""

    item_id: str = Field(..., description="Product identifier")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1) - how relevant this item is to user's preferences"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1) - how confident the model is in this recommendation"
    )
    item_price: Optional[float] = Field(
        None,
        ge=0.0,
        description="Item price in dollars"
    )
    recommendation_reason: str = Field(
        ...,
        description="Why this item was recommended"
    )
    model_source: str = Field(
        ...,
        description="Model that generated this recommendation (als, item_cf, popularity, blend)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "7003858505",
                "relevance_score": 0.87,
                "confidence_score": 0.92,
                "item_price": 4.99,
                "recommendation_reason": "Matches your preferences - fits your budget",
                "model_source": "blend"
            }
        }


class SimilarItem(BaseModel):
    """Similar item in similar items response"""

    item_id: str = Field(..., description="Product identifier")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity/relevance score (0-1)"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    item_price: Optional[float] = Field(
        None,
        ge=0.0,
        description="Item price in dollars"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "7003858505",
                "relevance_score": 0.85,
                "confidence_score": 0.78,
                "item_price": 4.99
            }
        }


class SimilarItemsResponse(BaseModel):
    """Response for similar items endpoint"""

    item_id: str = Field(..., description="Source item identifier")
    similar_items: List[SimilarItem] = Field(
        ...,
        description="List of similar items"
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to find similar items"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "7003858505",
                "similar_items": [
                    {
                        "item_id": "7003858506",
                        "relevance_score": 0.92,
                        "confidence_score": 0.85,
                        "item_price": 4.99
                    },
                    {
                        "item_id": "7003858507",
                        "relevance_score": 0.88,
                        "confidence_score": 0.80,
                        "item_price": 5.49
                    }
                ],
                "processing_time_ms": 12.5
            }
        }


class UserInfo(BaseModel):
    """User information returned with recommendations"""

    user_id: str = Field(..., description="User identifier")
    user_type: UserType = Field(..., description="User segment classification")
    spending_segment: Optional[str] = Field(
        None,
        description="User spending segment (small, low_average, average, high)"
    )
    total_purchases: int = Field(
        ...,
        ge=0,
        description="Total purchase count"
    )
    unique_items: int = Field(
        ...,
        ge=0,
        description="Number of unique items purchased"
    )
    avg_item_price: Optional[float] = Field(
        None,
        ge=0.0,
        description="Average item price"
    )
    last_purchase_date: Optional[datetime] = Field(
        None,
        description="Date of last purchase"
    )


class RecommendationResponse(BaseModel):
    """Complete recommendation response"""

    user_id: str = Field(..., description="User identifier")
    user_info: UserInfo = Field(..., description="User information")
    recommendations: List[RecommendationItem] = Field(
        ...,
        description="List of recommended products"
    )
    primary_model: ModelType = Field(
        ...,
        description="Primary model used for recommendations"
    )
    fallback_used: bool = Field(
        False,
        description="Whether fallback strategy was used"
    )
    processing_time_ms: float = Field(
        ...,
        description="Time taken to generate recommendations"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "41786230378",
                "user_info": {
                    "user_id": "41786230378",
                    "user_type": "loyal",
                    "spending_segment": "average",
                    "total_purchases": 1523,
                    "unique_items": 487,
                    "avg_item_price": 4.12,
                    "last_purchase_date": "2025-12-15T10:30:00"
                },
                "recommendations": [
                    {
                        "item_id": "7003858505",
                        "relevance_score": 0.87,
                        "confidence_score": 0.92,
                        "item_price": 4.99,
                        "recommendation_reason": "Matches your preferences - fits your budget",
                        "model_source": "blend"
                    }
                ],
                "primary_model": "hybrid",
                "fallback_used": False,
                "processing_time_ms": 45.2,
                "generated_at": "2025-12-20T14:30:00"
            }
        }


class UserRecommendationRequest(BaseModel):
    """Request for user recommendations"""

    user_id: str = Field(..., description="User identifier")
    num_recommendations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations"
    )
    model_type: Optional[str] = Field(
        None,
        description="Model to use: 'als', 'item_cf', or 'hybrid' (default)"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Current timestamp for repurchase cycle calculations"
    )
    exclude_purchased: bool = Field(
        True,
        description="Exclude already purchased items"
    )
    price_range_min: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum price filter"
    )
    price_range_max: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum price filter"
    )

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Validate user_id is not empty"""
        if not v or not v.strip():
            raise ValueError("user_id cannot be empty")
        return v.strip()

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate model_type is valid"""
        if v is None:
            return None
        v = v.lower().strip()
        # matrix_factorization is an alias for als
        valid_types = ['als', 'item_cf', 'hybrid', 'matrix_factorization']
        if v not in valid_types:
            raise ValueError(f"model_type must be one of: {valid_types}")
        return v


class SimilarItemsRequest(BaseModel):
    """Request for similar items"""

    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar items to return"
    )


class ModelStats(BaseModel):
    """Statistics for a recommendation model"""

    model_type: ModelType
    total_users: int
    total_items: int
    total_interactions: int
    sparsity: float
    last_trained: Optional[datetime]
    training_time_seconds: Optional[float]


class SystemStats(BaseModel):
    """System-wide statistics"""

    models: List[ModelStats]
    total_loyal_users: int
    total_new_users: int
    total_items: int
    cache_hit_rate: float
    avg_response_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    model_version: Optional[str] = Field(None, description="Version of loaded models")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
