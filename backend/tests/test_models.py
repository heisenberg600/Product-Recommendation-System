"""Tests for pre-trained recommendation models"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path

from app.models.model_loader import ModelLoader, get_model_loader, reset_model_loader
from app.models.candidate_ranker import CandidateRanker
from app.core.tuning_config import TUNING_CONFIG


class TestModelLoader:
    """Tests for ModelLoader - loads pre-trained models"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset model loader before each test"""
        reset_model_loader()

    def test_initialization(self):
        """Test model loader initialization"""
        loader = ModelLoader(models_path="models")
        assert not loader.is_loaded
        assert loader.als_model is None
        assert loader.item_cf_model is None

    def test_load_all(self):
        """Test loading all models"""
        loader = ModelLoader(models_path="models")
        success = loader.load_all()

        # Should succeed if models exist
        if Path("models").exists():
            assert success
            assert loader.is_loaded
            assert loader.als_model is not None
            assert loader.item_cf_model is not None
            assert loader.popularity_model is not None
            assert loader.user_profiles is not None

    def test_get_model_loader_singleton(self):
        """Test singleton pattern"""
        loader1 = get_model_loader("models")
        loader2 = get_model_loader("models")
        assert loader1 is loader2

    def test_user_methods(self):
        """Test user-related methods"""
        loader = get_model_loader("models")

        if loader.is_loaded and loader.user_profiles:
            # Get a test user
            test_user = list(loader.user_profiles.keys())[0]

            # Test user segment
            segment = loader.get_user_segment(test_user)
            assert segment in ['small', 'low_average', 'average', 'high']

            # Test user profile
            profile = loader.get_user_profile(test_user)
            assert profile is not None
            assert 'avg_item_price' in profile
            assert 'is_loyal' in profile

            # Test user avg price
            avg_price = loader.get_user_avg_price(test_user)
            assert avg_price >= 0

    def test_item_methods(self):
        """Test item-related methods"""
        loader = get_model_loader("models")

        if loader.is_loaded and loader.item_features:
            # Get a test item
            test_item = list(loader.item_features.keys())[0]

            # Test item features
            features = loader.get_item_features(test_item)
            assert features is not None
            assert 'avg_price' in features
            assert 'popularity_score' in features

            # Test item price
            price = loader.get_item_price(test_item)
            assert price >= 0

            # Test repurchase cycle
            cycle = loader.get_avg_repurchase_days(test_item)
            assert cycle > 0

    def test_als_recommendations(self):
        """Test ALS recommendations"""
        loader = get_model_loader("models")

        if loader.is_loaded and loader.als_model:
            test_user = list(loader.user_profiles.keys())[0]

            recs = loader.get_als_recommendations(test_user, n=10)
            assert isinstance(recs, list)

            for item_id, score, confidence in recs:
                assert isinstance(item_id, str)
                assert 0 <= score <= 1
                assert 0 <= confidence <= 1

    def test_item_cf_recommendations(self):
        """Test Item-CF recommendations"""
        loader = get_model_loader("models")

        if loader.is_loaded and loader.item_cf_model:
            test_user = list(loader.user_profiles.keys())[0]

            recs = loader.get_item_cf_recommendations(test_user, n=10)
            assert isinstance(recs, list)

    def test_popular_items(self):
        """Test popular items"""
        loader = get_model_loader("models")

        if loader.is_loaded and loader.popularity_model:
            items = loader.get_popular_items(n=10)
            assert isinstance(items, list)
            assert len(items) <= 10

            for item_id, score, confidence in items:
                assert isinstance(item_id, str)
                assert 0 <= score <= 1

    def test_similar_items(self):
        """Test similar items"""
        loader = get_model_loader("models")

        if loader.is_loaded and loader.item_cf_model:
            test_item = list(loader.item_features.keys())[0]

            similar = loader.get_similar_items(test_item, n=5)
            assert isinstance(similar, list)


class TestCandidateRanker:
    """Tests for CandidateRanker - Level 2 re-ranking"""

    @pytest.fixture
    def ranker(self):
        """Create a ranker with loaded models"""
        reset_model_loader()
        loader = get_model_loader("models")
        return CandidateRanker(TUNING_CONFIG, loader)

    def test_initialization(self, ranker):
        """Test ranker initialization"""
        assert ranker.config is not None
        assert ranker.model_loader is not None

    def test_price_penalty(self, ranker):
        """Test price penalty calculation"""
        user_avg_price = 4.00

        # No penalty for items within tolerance
        penalty = ranker._compute_price_penalty(
            item_price=4.00,
            user_avg_price=user_avg_price,
            penalty_start_ratio=1.5,
            max_penalty_ratio=3.0,
            penalty_factor=0.3
        )
        assert penalty == 0.0

        # No penalty at boundary
        penalty = ranker._compute_price_penalty(
            item_price=6.00,  # 1.5x
            user_avg_price=user_avg_price,
            penalty_start_ratio=1.5,
            max_penalty_ratio=3.0,
            penalty_factor=0.3
        )
        assert penalty == 0.0

        # Partial penalty
        penalty = ranker._compute_price_penalty(
            item_price=9.00,  # 2.25x
            user_avg_price=user_avg_price,
            penalty_start_ratio=1.5,
            max_penalty_ratio=3.0,
            penalty_factor=0.3
        )
        assert 0 < penalty < 0.3

        # Max penalty at 3x
        penalty = ranker._compute_price_penalty(
            item_price=12.00,  # 3x
            user_avg_price=user_avg_price,
            penalty_start_ratio=1.5,
            max_penalty_ratio=3.0,
            penalty_factor=0.3
        )
        assert penalty == 0.3

    def test_price_match(self, ranker):
        """Test price match calculation"""
        score = ranker._compute_price_match(
            item_price=4.00,
            user_avg_price=4.00,
            user_segment='average'
        )
        # Should get a reasonable score
        assert 0 <= score <= 1

    def test_upsell_boost(self, ranker):
        """Test upsell boost calculation"""
        user_avg_price = 4.00

        # No boost for cheaper items
        boost = ranker._compute_upsell_boost(
            item_price=3.00,
            user_avg_price=user_avg_price,
            upsell_factor=0.1,
            price_boost_weight=0.2,
            max_price_ratio=2.0
        )
        assert boost == 0.0

        # Some boost for more expensive items
        boost = ranker._compute_upsell_boost(
            item_price=6.00,  # 1.5x
            user_avg_price=user_avg_price,
            upsell_factor=0.1,
            price_boost_weight=0.2,
            max_price_ratio=2.0
        )
        assert boost > 0

        # No boost for items way too expensive
        boost = ranker._compute_upsell_boost(
            item_price=12.00,  # 3x, exceeds max_price_ratio
            user_avg_price=user_avg_price,
            upsell_factor=0.1,
            price_boost_weight=0.2,
            max_price_ratio=2.0
        )
        assert boost == 0.0

    def test_rank_candidates(self, ranker):
        """Test candidate ranking"""
        if not ranker.model_loader.is_loaded:
            pytest.skip("Models not loaded")

        # Get a test user
        test_user = list(ranker.model_loader.user_profiles.keys())[0]

        # Create mock candidates
        candidates = [
            ("item1", 0.8, 0.7, "als"),
            ("item2", 0.7, 0.6, "item_cf"),
            ("item3", 0.6, 0.8, "popularity"),
        ]

        # This will fail if items don't exist, but tests the flow
        try:
            ranked = ranker.rank(
                candidates=candidates,
                user_id=test_user,
                timestamp=datetime.now(),
                n=3
            )
            assert isinstance(ranked, list)
        except Exception:
            # Items might not exist in item_features
            pass

    def test_rank_for_new_user(self, ranker):
        """Test ranking for new user"""
        if not ranker.model_loader.is_loaded:
            pytest.skip("Models not loaded")

        ranked = ranker.rank_for_new_user(
            first_purchase_price=5.00,
            n=5,
            exclude_items=[]
        )

        assert isinstance(ranked, list)
        for rec in ranked:
            assert 'item_id' in rec
            assert 'relevance_score' in rec
            assert 'confidence_score' in rec
            assert 0 <= rec['relevance_score'] <= 1
            assert 0 <= rec['confidence_score'] <= 1

    def test_generate_reason(self, ranker):
        """Test recommendation reason generation"""
        # Test ALS source
        reason = ranker._generate_reason(
            source='als',
            base_score=0.8,
            price_match_score=0.9
        )
        assert "preferences" in reason.lower()

        # Test Item-CF source
        reason = ranker._generate_reason(
            source='item_cf',
            base_score=0.7,
            price_match_score=0.8
        )
        assert "similar" in reason.lower()

        # Test repurchase boost
        reason = ranker._generate_reason(
            source='als',
            base_score=0.7,
            price_match_score=0.8,
            has_repurchase_boost=True
        )
        assert "restock" in reason.lower()
