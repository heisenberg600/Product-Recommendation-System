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
        mock.data_loaded = True
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
        assert data["data_loaded"] is True


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
                total_purchases=100,
                unique_items=50,
                avg_item_price=4.99,
                last_purchase_date=datetime.now()
            ),
            recommendations=[
                RecommendationItem(
                    item_id="item_1",
                    relevance_score=0.95,
                    confidence=0.90,
                    item_price=4.99,
                    recommendation_reason="Similar to your purchases",
                    model_used=ModelType.HYBRID
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

    def test_get_recommendations_with_model_param(self, client, mock_service):
        """Test recommendations with specific model"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.LOYAL,
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

        # Verify correct model was passed
        mock_service.get_recommendations.assert_called_once()
        call_args = mock_service.get_recommendations.call_args
        assert call_args.kwargs["model_type"] == ModelType.ITEM_CF

    def test_get_recommendations_with_price_filter(self, client, mock_service):
        """Test recommendations with price filter"""
        mock_response = RecommendationResponse(
            user_id="test_user",
            user_info=UserInfo(
                user_id="test_user",
                user_type=UserType.NEW,
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


class TestSimilarItemsEndpoint:
    """Tests for similar items endpoint"""

    def test_get_similar_items_success(self, client, mock_service):
        """Test successful similar items request"""
        mock_service.get_similar_items.return_value = [
            {"item_id": "item_2", "similarity_score": 0.95},
            {"item_id": "item_3", "similarity_score": 0.85}
        ]

        response = client.get("/api/v1/items/item_1/similar?n=5")
        assert response.status_code == 200

        data = response.json()
        assert data["item_id"] == "item_1"
        assert len(data["similar_items"]) == 2

    def test_get_similar_items_not_found(self, client, mock_service):
        """Test when no similar items found"""
        mock_service.get_similar_items.return_value = []

        response = client.get("/api/v1/items/unknown_item/similar")
        assert response.status_code == 404


class TestPopularItemsEndpoint:
    """Tests for popular items endpoint"""

    def test_get_popular_items(self, client, mock_service):
        """Test getting popular items"""
        mock_service.get_popular_items.return_value = [
            {"item_id": "item_1", "purchase_count": 1000},
            {"item_id": "item_2", "purchase_count": 500}
        ]

        response = client.get("/api/v1/items/popular?n=10")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["items"]) == 2


class TestUsersEndpoint:
    """Tests for users endpoints"""

    def test_get_all_users(self, client, mock_service):
        """Test getting all users"""
        mock_service.get_all_users.return_value = {
            "loyal": ["user_1", "user_2"],
            "new": ["user_3"]
        }

        response = client.get("/api/v1/users")
        assert response.status_code == 200

        data = response.json()
        assert data["loyal_count"] == 2
        assert data["new_count"] == 1


class TestModelsEndpoint:
    """Tests for models endpoint"""

    def test_get_available_models(self, client, mock_service):
        """Test getting available models"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert len(data["models"]) == 5

        model_types = [m["type"] for m in data["models"]]
        assert "hybrid" in model_types
        assert "item_cf" in model_types
        assert "popularity" in model_types


class TestStatsEndpoint:
    """Tests for statistics endpoint"""

    def test_get_statistics(self, client, mock_service):
        """Test getting system statistics"""
        mock_service.get_statistics.return_value = {
            "is_initialized": True,
            "cache_size": 10,
            "data_stats": {},
            "model_stats": {}
        }

        response = client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["is_initialized"] is True
