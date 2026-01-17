"""Recommendation models - Pre-trained model architecture"""

from app.models.model_loader import ModelLoader, get_model_loader, reset_model_loader
from app.models.candidate_ranker import CandidateRanker

__all__ = [
    "ModelLoader",
    "get_model_loader",
    "reset_model_loader",
    "CandidateRanker",
]
