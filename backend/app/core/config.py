"""Application configuration and settings"""

import os
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

    # Server - Railway provides PORT env var
    port: int = int(os.getenv("PORT", "8000"))

    # API
    api_prefix: str = "/api/v1"

    # CORS - set CORS_ORIGINS env var as comma-separated list for production
    # e.g., CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
    @property
    def cors_origins(self) -> list[str]:
        env_origins = os.getenv("CORS_ORIGINS", "")
        if env_origins:
            return [origin.strip() for origin in env_origins.split(",")]
        # Default for local development
        return ["http://localhost:3000", "http://localhost:5173"]

    # Data paths - check multiple locations for deployment flexibility
    @property
    def data_dir(self) -> Path:
        # Check environment variable first
        env_path = os.getenv("DATA_DIR")
        if env_path:
            return Path(env_path)
        # Try relative to config file (development)
        config_relative = Path(__file__).parent.parent.parent.parent / "data"
        if config_relative.exists():
            return config_relative
        # Try relative to CWD (Railway deployment)
        cwd_relative = Path("../data")
        if cwd_relative.exists():
            return cwd_relative
        return config_relative  # Default

    @property
    def models_dir(self) -> Path:
        # Check environment variable first
        env_path = os.getenv("MODELS_DIR")
        if env_path:
            return Path(env_path)
        # Try relative to config file (development)
        config_relative = Path(__file__).parent.parent.parent.parent / "models"
        if config_relative.exists():
            return config_relative
        # Try relative to CWD (Railway deployment)
        cwd_relative = Path("../models")
        if cwd_relative.exists():
            return cwd_relative
        return config_relative  # Default

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
