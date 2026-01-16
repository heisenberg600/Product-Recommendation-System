"""Recommendation models"""

from app.models.base import BaseRecommender
from app.models.item_cf import ItemCFRecommender
from app.models.matrix_factorization import MatrixFactorizationRecommender
from app.models.popularity import PopularityRecommender
from app.models.price_segment import PriceSegmentRecommender
from app.models.hybrid import HybridRecommender

__all__ = [
    "BaseRecommender",
    "ItemCFRecommender",
    "MatrixFactorizationRecommender",
    "PopularityRecommender",
    "PriceSegmentRecommender",
    "HybridRecommender",
]
