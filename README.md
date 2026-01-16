# Product Recommendation System

A production-grade, AI-powered recommendation engine that personalizes product suggestions based on user behavior and purchase history.

## Features

- **Multiple Recommendation Models**:
  - **Item-Item Collaborative Filtering**: Based on co-purchase patterns (Amazon-style)
  - **Matrix Factorization (ALS)**: Learns latent user/item factors for implicit feedback
  - **Popularity-based**: Trending and bestselling items for cold-start
  - **Price Segment**: Matches user's price preferences
  - **Hybrid**: Combines all models for optimal results

- **User Segmentation**:
  - **Loyal Customers**: Rich history enables personalized CF recommendations
  - **New Customers**: Cold-start handling with popularity and clustering

- **Production-Ready Architecture**:
  - Type-safe FastAPI backend with Pydantic validation
  - React + TypeScript frontend with TailwindCSS
  - Comprehensive test suite
  - Caching and performance optimization

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/api/v1/health

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at http://localhost:5173

## API Endpoints

### Recommendations

```bash
# Get recommendations for a user
GET /api/v1/recommendations/{user_id}?n=5&model=hybrid

# Get recommendations with advanced options (POST)
POST /api/v1/recommendations
{
    "user_id": "41786230378",
    "num_recommendations": 5,
    "model_type": "item_cf",
    "exclude_purchased": true,
    "price_range_min": 5,
    "price_range_max": 20
}
```

### Items

```bash
# Get similar items
GET /api/v1/items/{item_id}/similar?n=10

# Get popular items
GET /api/v1/items/popular?n=10
```

### System

```bash
# Health check
GET /api/v1/health

# Available models
GET /api/v1/models

# System statistics
GET /api/v1/stats

# All users
GET /api/v1/users
```

## Response Format

```json
{
    "user_id": "41786230378",
    "user_info": {
        "user_id": "41786230378",
        "user_type": "loyal",
        "total_purchases": 1523,
        "unique_items": 487,
        "avg_item_price": 4.12
    },
    "recommendations": [
        {
            "item_id": "7003858505",
            "relevance_score": 0.87,
            "confidence": 0.92,
            "item_price": 4.99,
            "recommendation_reason": "Similar to 5 items in your purchase history",
            "model_used": "item_cf"
        }
    ],
    "primary_model": "hybrid",
    "fallback_used": false,
    "processing_time_ms": 45.2
}
```

## Project Structure

```
Product-Recommendation-System/
├── backend/
│   ├── app/
│   │   ├── api/           # API routes
│   │   ├── core/          # Configuration
│   │   ├── models/        # Recommendation models
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   └── utils/         # Data loading/cleaning
│   ├── tests/             # Test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom hooks
│   │   ├── api/           # API client
│   │   └── types/         # TypeScript types
│   └── package.json
├── notebooks/
│   └── data_analysis.ipynb
├── data/
│   └── Data Science - Assignment.xlsx
└── research/
    └── deep_research.txt
```

## Models Overview

### 1. Item-Item Collaborative Filtering
- Computes item similarities based on co-purchase patterns
- Recommends items similar to user's purchase history
- Best for: Loyal customers with substantial history

### 2. Matrix Factorization (ALS)
- Learns latent factors from user-item interactions
- Handles implicit feedback (purchases) effectively
- Best for: Users with diverse purchase history

### 3. Popularity-based
- Recommends trending and bestselling items
- Time-weighted for recency bias
- Best for: New customers (cold-start)

### 4. Price Segment
- Creates price segments and matches user preferences
- Ensures price-relevant recommendations
- Best for: Price-sensitive recommendations

### 5. Hybrid
- Combines all models with dynamic weighting
- Adjusts weights based on user type and history size
- Best for: All users (default choice)

## Running Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

## Data Analysis

Run the Jupyter notebook for exploratory data analysis:

```bash
cd notebooks
jupyter notebook data_analysis.ipynb
```

## Configuration

Environment variables (create `.env` in backend/):

```env
DEBUG=false
ENVIRONMENT=development
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
CACHE_TTL_SECONDS=3600
```

## Performance

- Target latency: <100ms for recommendations
- Caching: TTL-based caching for repeated requests
- Precomputation: Item similarities computed offline

## License

MIT License
