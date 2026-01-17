"""Application configuration and settings"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with type-safe configuration"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Product Recommendation System"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Data paths
    data_dir: Path = Path(__file__).parent.parent.parent.parent / "data"
    models_dir: Path = Path(__file__).parent.parent.parent.parent / "models"
    excel_file: str = "Data Science - Assignment.xlsx"

    # Model parameters
    num_recommendations: int = 5
    min_support_threshold: int = 2
    similarity_top_k: int = 50

    # User classification thresholds
    loyal_user_min_purchases: int = 10

    # Cache settings
    cache_ttl_seconds: int = 3600

    # Matrix Factorization parameters
    mf_factors: int = 50
    mf_iterations: int = 15
    mf_regularization: float = 0.01

    @property
    def excel_path(self) -> Path:
        return self.data_dir / self.excel_file


settings = Settings()
