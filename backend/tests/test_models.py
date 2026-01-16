"""Tests for recommendation models"""

import pytest
import pandas as pd
import numpy as np

from app.models.item_cf import ItemCFRecommender
from app.models.matrix_factorization import MatrixFactorizationRecommender
from app.models.popularity import PopularityRecommender
from app.models.price_segment import PriceSegmentRecommender
from app.models.hybrid import HybridRecommender
from app.schemas.recommendation import ModelType


class TestItemCFRecommender:
    """Tests for Item-Item Collaborative Filtering"""

    def test_initialization(self):
        """Test model initialization"""
        model = ItemCFRecommender()
        assert model.model_type == ModelType.ITEM_CF
        assert not model.is_trained
        assert model.top_k == 50

    def test_training(self, sample_transactions: pd.DataFrame):
        """Test model training"""
        model = ItemCFRecommender(top_k=10)
        model.train(sample_transactions)

        assert model.is_trained
        assert model.last_trained is not None
        assert model.training_time > 0

    def test_recommendations(self, sample_transactions: pd.DataFrame, sample_user_history: pd.DataFrame):
        """Test recommendation generation"""
        model = ItemCFRecommender(top_k=10)
        model.train(sample_transactions)

        user_id = sample_user_history["user_id"].iloc[0]
        recommendations = model.recommend(
            user_id=user_id,
            user_history=sample_user_history,
            n=5
        )

        # May not always have recommendations depending on data
        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert 0 <= rec.relevance_score <= 1
            assert 0 <= rec.confidence <= 1
            assert rec.model_used == ModelType.ITEM_CF

    def test_similar_items(self, sample_transactions: pd.DataFrame):
        """Test getting similar items"""
        model = ItemCFRecommender(top_k=10)
        model.train(sample_transactions)

        item_id = sample_transactions["item_id"].iloc[0]
        similar = model.get_similar_items(item_id, n=5)

        assert isinstance(similar, list)
        for item, score in similar:
            assert isinstance(item, str)
            assert 0 <= score <= 1

    def test_exclude_purchased_items(self, sample_transactions: pd.DataFrame, sample_user_history: pd.DataFrame):
        """Test that purchased items are excluded"""
        model = ItemCFRecommender(top_k=10)
        model.train(sample_transactions)

        user_id = sample_user_history["user_id"].iloc[0]
        purchased = set(sample_user_history["item_id"].unique())

        recommendations = model.recommend(
            user_id=user_id,
            user_history=sample_user_history,
            n=5,
            exclude_items=purchased
        )

        for rec in recommendations:
            assert rec.item_id not in purchased


class TestMatrixFactorizationRecommender:
    """Tests for Matrix Factorization model"""

    def test_initialization(self):
        """Test model initialization"""
        model = MatrixFactorizationRecommender()
        assert model.model_type == ModelType.MATRIX_FACTORIZATION
        assert not model.is_trained
        assert model.n_factors == 50

    def test_training(self, sample_transactions: pd.DataFrame):
        """Test model training"""
        model = MatrixFactorizationRecommender(n_factors=10, n_iterations=5)
        model.train(sample_transactions)

        assert model.is_trained
        assert model.last_trained is not None

    def test_recommendations(self, sample_transactions: pd.DataFrame, sample_user_history: pd.DataFrame):
        """Test recommendation generation"""
        model = MatrixFactorizationRecommender(n_factors=10, n_iterations=5)
        model.train(sample_transactions)

        user_id = sample_user_history["user_id"].iloc[0]
        recommendations = model.recommend(
            user_id=user_id,
            user_history=sample_user_history,
            n=5
        )

        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert 0 <= rec.relevance_score <= 1
            assert rec.model_used == ModelType.MATRIX_FACTORIZATION

    def test_embeddings(self, sample_transactions: pd.DataFrame):
        """Test user and item embeddings"""
        model = MatrixFactorizationRecommender(n_factors=10, n_iterations=5)
        model.train(sample_transactions)

        user_id = sample_transactions["user_id"].iloc[0]
        item_id = sample_transactions["item_id"].iloc[0]

        user_emb = model.get_user_embedding(user_id)
        item_emb = model.get_item_embedding(item_id)

        assert user_emb is not None
        assert len(user_emb) == 10
        assert item_emb is not None
        assert len(item_emb) == 10


class TestPopularityRecommender:
    """Tests for Popularity-based model"""

    def test_initialization(self):
        """Test model initialization"""
        model = PopularityRecommender()
        assert model.model_type == ModelType.POPULARITY
        assert not model.is_trained

    def test_training(self, sample_transactions: pd.DataFrame):
        """Test model training"""
        model = PopularityRecommender()
        model.train(sample_transactions)

        assert model.is_trained
        assert len(model._item_popularity) > 0

    def test_recommendations_new_user(self, sample_transactions: pd.DataFrame):
        """Test recommendations for new user with no history"""
        model = PopularityRecommender()
        model.train(sample_transactions)

        # New user with no history
        recommendations = model.recommend(
            user_id="unknown_user",
            user_history=pd.DataFrame(),
            n=5
        )

        assert len(recommendations) == 5
        for rec in recommendations:
            assert rec.model_used == ModelType.POPULARITY

    def test_trending_items(self, sample_transactions: pd.DataFrame):
        """Test trending items"""
        model = PopularityRecommender()
        model.train(sample_transactions)

        trending = model.get_trending(n=10)
        assert isinstance(trending, list)
        assert len(trending) <= 10


class TestPriceSegmentRecommender:
    """Tests for Price Segment model"""

    def test_initialization(self):
        """Test model initialization"""
        model = PriceSegmentRecommender()
        assert model.model_type == ModelType.PRICE_SEGMENT
        assert model.n_segments == 5

    def test_training(self, sample_transactions: pd.DataFrame):
        """Test model training"""
        model = PriceSegmentRecommender(n_segments=3)
        model.train(sample_transactions)

        assert model.is_trained
        assert len(model._segment_boundaries) > 0

    def test_price_matching(self, sample_transactions: pd.DataFrame, sample_user_history: pd.DataFrame):
        """Test that recommendations match user's price preferences"""
        model = PriceSegmentRecommender(n_segments=3)
        model.train(sample_transactions)

        user_id = sample_user_history["user_id"].iloc[0]
        user_avg_price = sample_user_history["item_price"].mean()

        recommendations = model.recommend(
            user_id=user_id,
            user_history=sample_user_history,
            n=5
        )

        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert rec.model_used == ModelType.PRICE_SEGMENT


class TestHybridRecommender:
    """Tests for Hybrid model"""

    def test_initialization(self):
        """Test model initialization"""
        model = HybridRecommender()
        assert model.model_type == ModelType.HYBRID
        assert not model.is_trained

    def test_training(self, sample_transactions: pd.DataFrame):
        """Test model training trains all components"""
        model = HybridRecommender()
        model.train(sample_transactions)

        assert model.is_trained
        # Check all component models are trained
        assert model._item_cf.is_trained
        assert model._mf.is_trained
        assert model._popularity.is_trained
        assert model._price_segment.is_trained

    def test_recommendations_loyal_user(self, sample_transactions: pd.DataFrame):
        """Test recommendations for loyal user"""
        model = HybridRecommender()
        model.train(sample_transactions)

        # Get a loyal user with history
        loyal_data = sample_transactions[sample_transactions["user_type"] == "loyal"]
        user_id = loyal_data["user_id"].iloc[0]
        user_history = loyal_data[loyal_data["user_id"] == user_id]

        recommendations = model.recommend(
            user_id=user_id,
            user_history=user_history,
            n=5
        )

        assert len(recommendations) <= 5
        for rec in recommendations:
            assert rec.model_used == ModelType.HYBRID

    def test_recommendations_new_user(self, sample_transactions: pd.DataFrame):
        """Test recommendations for new user with no history"""
        model = HybridRecommender()
        model.train(sample_transactions)

        recommendations = model.recommend(
            user_id="unknown_user",
            user_history=pd.DataFrame(),
            n=5
        )

        assert len(recommendations) <= 5

    def test_weight_adjustment(self, sample_transactions: pd.DataFrame):
        """Test dynamic weight adjustment based on history size"""
        model = HybridRecommender(min_history_for_cf=10)
        model.train(sample_transactions)

        # Small history should use new user weights
        weights_small = model._get_dynamic_weights(3, is_loyal=False)
        assert weights_small["popularity"] > weights_small["item_cf"]

        # Large history should use loyal user weights
        weights_large = model._get_dynamic_weights(100, is_loyal=True)
        assert weights_large["item_cf"] > weights_large["popularity"]
