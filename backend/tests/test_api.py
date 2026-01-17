"""Tests for API endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.main import app
from app.schemas.recommendation import (
    RecommendationResponse,
    RecommendationItem,
    UserInfo,
    UserType,
    ModelType,
    HealthResponse,
    SimilarItemsResponse,
    SimilarItem,
)


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """Mock recommendation service"""
    with patch("app.api.routes.recommendation_service") as mock:
        mock.is_initialized = True
        mock._model_loader = MagicMock()
        mock._model_loader.get_model_version.return_value = "1.0.0"
        yield mock


class TestHealthEndpoint:
    """Tests for health check endpoint"""

    def test_health_check(self, client, mock_service):
        """Test health check returns correct status"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert data["model_version"] == "1.0.0"


class TestRecommendationsEndpoint:
    """Tests for recommendations endpoint"""

    def test_get_recommendations_success(self, client, mock_service):
        """Test successful recommendations request"""
        # Mock the service response
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
                spending_segment="average",
                total_purchases=100,
                unique_items=50,
                avg_item_price=4.99,
                last_purchase_date=datetime.now()
            ),
            recommendations=[
                RecommendationItem(
                    item_id="item_1",
                    relevance_score=0.95,
                    confidence_score=0.90,
                    item_price=4.99,
                    recommendation_reason="Matches your preferences",
                    model_source="hybrid"
                )
            ],
            primary_model=ModelType.HYBRID,
            fallback_used=False,
            processing_time_ms=45.2
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.get("/api/v1/recommendations/test_user?n=5")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == "test_user"
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["item_id"] == "item_1"
        assert data["recommendations"][0]["relevance_score"] == 0.95
        assert data["recommendations"][0]["confidence_score"] == 0.90
        assert data["recommendations"][0]["model_source"] == "hybrid"
        assert data["user_info"]["spending_segment"] == "average"

    def test_get_recommendations_with_als_model(self, client, mock_service):
        """Test recommendations with ALS model"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
                spending_segment="high",
                total_purchases=100,
                unique_items=50
            ),
            recommendations=[],
            primary_model=ModelType.ALS,
            fallback_used=False,
            processing_time_ms=30.0
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.get("/api/v1/recommendations/test_user?model=als")
        assert response.status_code == 200

        # Verify correct model was passed
        mock_service.get_recommendations.assert_called_once()
        call_args = mock_service.get_recommendations.call_args
        assert call_args.kwargs["model_type"] == "als"

    def test_get_recommendations_with_item_cf_model(self, client, mock_service):
        """Test recommendations with Item-CF model"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
                spending_segment="average",
                total_purchases=100,
                unique_items=50
            ),
            recommendations=[],
            primary_model=ModelType.ITEM_CF,
            fallback_used=False,
            processing_time_ms=30.0
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.get("/api/v1/recommendations/test_user?model=item_cf")
        assert response.status_code == 200

        call_args = mock_service.get_recommendations.call_args
        assert call_args.kwargs["model_type"] == "item_cf"

    def test_get_recommendations_with_timestamp(self, client, mock_service):
        """Test recommendations with timestamp parameter"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
                spending_segment="average",
                total_purchases=100,
                unique_items=50
            ),
            recommendations=[],
            primary_model=ModelType.HYBRID,
            fallback_used=False,
            processing_time_ms=30.0
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.get(
            "/api/v1/recommendations/test_user?timestamp=2025-01-15T10:00:00"
        )
        assert response.status_code == 200

        call_args = mock_service.get_recommendations.call_args
        assert call_args.kwargs["timestamp"] is not None

    def test_get_recommendations_with_price_filter(self, client, mock_service):
        """Test recommendations with price filter"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.NEW,
                spending_segment="small",
                total_purchases=5,
                unique_items=3
            ),
            recommendations=[],
            primary_model=ModelType.HYBRID,
            fallback_used=False,
            processing_time_ms=25.0
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.get("/api/v1/recommendations/test_user?price_min=5&price_max=20")
        assert response.status_code == 200

        # Verify price filter was passed
        call_args = mock_service.get_recommendations.call_args
        assert call_args.kwargs["price_range_min"] == 5.0
        assert call_args.kwargs["price_range_max"] == 20.0

    def test_get_recommendations_service_not_ready(self, client, mock_service):
        """Test error when service is not initialized"""
        mock_service.is_initialized = False

        response = client.get("/api/v1/recommendations/test_user")
        assert response.status_code == 503

    def test_post_recommendations(self, client, mock_service):
        """Test POST endpoint for recommendations"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
                spending_segment="average",
                total_purchases=100,
                unique_items=50
            ),
            recommendations=[],
            primary_model=ModelType.HYBRID,
            fallback_used=False,
            processing_time_ms=30.0
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.post(
            "/api/v1/recommendations",
            json={
                "user_id": "test_user",
                "num_recommendations": 5,
                "exclude_purchased": True
            }
        )
        assert response.status_code == 200

    def test_post_recommendations_with_model_type(self, client, mock_service):
        """Test POST endpoint with model_type"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
                spending_segment="average",
                total_purchases=100,
                unique_items=50
            ),
            recommendations=[],
            primary_model=ModelType.ALS,
            fallback_used=False,
            processing_time_ms=30.0
        )
        mock_service.get_recommendations.return_value = mock_response

        response = client.post(
            "/api/v1/recommendations",
            json={
                "user_id": "test_user",
                "num_recommendations": 5,
                "model_type": "als"
            }
        )
        assert response.status_code == 200

        call_args = mock_service.get_recommendations.call_args
        assert call_args.kwargs["model_type"] == "als"


class TestSimilarItemsEndpoint:
    """Tests for similar items endpoint"""

    def test_get_similar_items_success(self, client, mock_service):
        """Test successful similar items request"""
        mock_response = SimilarItemsResponse(
            item_id="item_1",
            similar_items=[
                SimilarItem(
                    item_id="item_2",
                    relevance_score=0.95,
                    confidence_score=0.85,
                    item_price=4.99
                ),
                SimilarItem(
                    item_id="item_3",
                    relevance_score=0.85,
                    confidence_score=0.80,
                    item_price=5.99
                )
            ],
            processing_time_ms=12.5
        )
        mock_service.get_similar_items.return_value = mock_response

        response = client.get("/api/v1/items/item_1/similar?n=5")
        assert response.status_code == 200

        data = response.json()
        assert data["item_id"] == "item_1"
        assert len(data["similar_items"]) == 2
        assert data["similar_items"][0]["relevance_score"] == 0.95
        assert data["similar_items"][0]["confidence_score"] == 0.85

    def test_get_similar_items_not_found(self, client, mock_service):
        """Test when no similar items found"""
        mock_response = SimilarItemsResponse(
            item_id="unknown_item",
            similar_items=[],
            processing_time_ms=5.0
        )
        mock_service.get_similar_items.return_value = mock_response

        response = client.get("/api/v1/items/unknown_item/similar")
        assert response.status_code == 404


class TestPopularItemsEndpoint:
    """Tests for popular items endpoint"""

    def test_get_popular_items(self, client, mock_service):
        """Test getting popular items"""
        mock_service.get_popular_items.return_value = [
            {
                "item_id": "item_1",
                "item_price": 4.99,
                "popularity_score": 0.95,
                "confidence_score": 0.90,
                "purchase_count": 1000,
                "unique_buyers": 500
            },
            {
                "item_id": "item_2",
                "item_price": 5.99,
                "popularity_score": 0.85,
                "confidence_score": 0.85,
                "purchase_count": 500,
                "unique_buyers": 250
            }
        ]

        response = client.get("/api/v1/items/popular?n=10")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["items"]) == 2
        assert data["items"][0]["popularity_score"] == 0.95


class TestUserProfileEndpoint:
    """Tests for user profile endpoint"""

    def test_get_user_profile(self, client, mock_service):
        """Test getting user profile"""
        mock_service._get_user_info.return_value = UserInfo(
            user_id="test_user",
            user_type=UserType.LOYAL,
            spending_segment="high",
            total_purchases=100,
            unique_items=50,
            avg_item_price=4.99,
            last_purchase_date=datetime.now()
        )
        mock_service._model_loader.get_user_profile.return_value = {
            "avg_item_price": 4.99,
            "total_purchases": 100,
            "segment": "high"
        }

        response = client.get("/api/v1/users/test_user/profile")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["is_known_user"] is True


class TestConfigEndpoint:
    """Tests for tuning configuration endpoint"""

    def test_get_tuning_config(self, client, mock_service):
        """Test getting tuning configuration"""
        response = client.get("/api/v1/config")
        assert response.status_code == 200

        data = response.json()
        assert "config" in data
        assert "description" in data
        assert "candidate_pool_size" in data["config"]
        assert "model_weights" in data["config"]
        assert "upsell" in data["config"]


class TestModelsEndpoint:
    """Tests for models endpoint"""

    def test_get_available_models(self, client, mock_service):
        """Test getting available models"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "architecture" in data
        assert len(data["models"]) == 4  # hybrid, als, item_cf, popularity

        model_types = [m["type"] for m in data["models"]]
        assert "hybrid" in model_types
        assert "als" in model_types
        assert "item_cf" in model_types
        assert "popularity" in model_types


class TestStatsEndpoint:
    """Tests for statistics endpoint"""

    def test_get_statistics(self, client, mock_service):
        """Test getting system statistics"""
        mock_service.get_statistics.return_value = {
            "is_initialized": True,
            "model_load_time": "2025-01-15T10:00:00",
            "cache_size": 10,
            "cache_max_size": 1000,
            "model_stats": {
                "is_loaded": True,
                "version": "1.0.0",
                "n_users": 1000,
                "n_items": 500
            },
            "config": {
                "candidate_pool_size": 200,
                "model_weights": {"als": 0.4, "item_cf": 0.4, "popularity": 0.2}
            }
        }

        response = client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["is_initialized"] is True
        assert "model_stats" in data
        assert "config" in data
