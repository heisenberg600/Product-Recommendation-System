"""FastAPI application entry point"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.recommendation_service import recommendation_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    Initializes services on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting Product Recommendation System...")
    setup_logging()

    # Initialize recommendation service
    success = await recommendation_service.initialize()
    if success:
        logger.info("Recommendation service initialized successfully")
    else:
        logger.error("Failed to initialize recommendation service")

    yield

    # Shutdown
    logger.info("Shutting down Product Recommendation System...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## Product Recommendation System API

A production-grade recommendation engine that personalizes product suggestions
based on user behavior and purchase history.

### Features:
- **Multiple recommendation models**: Item-CF, Matrix Factorization, Popularity, Hybrid
- **User segmentation**: Different strategies for loyal vs. new customers
- **Cold-start handling**: Fallback strategies for new users
- **Price filtering**: Filter recommendations by price range
- **High performance**: Sub-100ms response times with caching

### User Segments:
1. **Loyal Customers**: Users with established purchase history receive personalized
   recommendations based on collaborative filtering and matrix factorization.

2. **New Customers**: Users with limited history receive popularity-based and
   price-segment recommendations.

### API Usage:
```python
import requests

# Get recommendations for a user
response = requests.get(
    "http://localhost:8000/api/v1/recommendations/41786230378",
    params={"n": 5}
)
recommendations = response.json()
```
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=settings.api_prefix)


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict:
    """
    Root endpoint with API information.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "recommendations": f"{settings.api_prefix}/recommendations/{{user_id}}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug
    )
