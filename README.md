# Product Recommendation System

A production-grade, AI-powered recommendation engine that personalizes product suggestions based on user behavior, purchase history, and spending patterns.

## Features

### Recommendation Models (Selectable)

- **ALS (Matrix Factorization)** - Learns latent user/item factors using implicit feedback signals (spending ratio, quantity ratio). Best for users with diverse purchase history.
- **Item-Item Collaborative Filtering** - Based on co-purchase patterns with spending-weighted similarity. Best for loyal customers with substantial history.
- **Hybrid (Default)** - Weighted blend of all models:
  - ALS: 40%
  - Item-CF: 40%
  - Popularity: 20%

- **Popularity-based** - Trending items weighted by purchase count, unique buyers, and total quantity. Used for cold-start scenarios.

### Two-Level Recommendation Architecture

1. **Level 1 - Candidate Generation**: Fetches 200 candidate items from models
2. **Level 2 - Re-ranking**: Applies:
   - User spending segment filtering
   - Upsell boost for higher-priced items
   - Repurchase cycle exclusions
   - Price sensitivity adjustments

### User Segmentation

Users are categorized by spending behavior:
- **Small** (bottom 25%) - Lowest spenders
- **Low Average** (25-50%) - Below median
- **Average** (50-75%) - Above median
- **High** (top 25%) - Top spenders

### Implicit Feedback Signals

- **Spending Ratio**: `item_price / user_avg_item_price` (>1 = spent more than usual)
- **Quantity Ratio**: `units_sold / item_avg_quantity` (>1 = bought more than usual)

### Additional Features

- **Similar Items**: Find top 5 similar items for any product
- **Repurchase Cycle Tracking**: Exclude recently purchased items based on typical repurchase patterns
- **Upsell Knobs**: Configurable boost for higher-priced items to increase revenue
- **Price Filtering**: Filter recommendations by min/max price range
- **Offline Model Training**: Models are pre-trained and serialized for fast loading

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

### Train Models (First Time Setup)

```bash
cd scripts

# Train all models and generate serialized files
python train_models.py --data-path ../data --output-path ../models
```

This generates:
- `models/als_model.pkl` - ALS user/item factors
- `models/item_cf_similarities.pkl` - Item similarity matrix
- `models/user_profiles.pkl` - User segments and profiles
- `models/item_features.pkl` - Item prices and popularity
- `models/repurchase_cycles.pkl` - Item repurchase patterns
- `models/metadata.json` - Training info and version

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
# Get recommendations for a user (with model selection)
GET /api/v1/recommendations/{user_id}?n=5&model=hybrid&timestamp=2025-01-15T10:00:00

# Model options: als, item_cf, hybrid (default)

# Get recommendations with advanced options (POST)
POST /api/v1/recommendations
{
    "user_id": "41786230378",
    "num_recommendations": 5,
    "model_type": "als",
    "exclude_purchased": true,
    "price_range_min": 5,
    "price_range_max": 20,
    "timestamp": "2025-01-15T10:00:00"
}
```

### Similar Items

```bash
# Get similar items for a product
GET /api/v1/items/{item_id}/similar?n=5

# Response:
{
    "item_id": "123",
    "similar_items": [
        {
            "item_id": "456",
            "relevance_score": 0.92,
            "confidence_score": 0.85,
            "item_price": 4.99
        }
    ],
    "processing_time_ms": 12.5
}
```

### Items

```bash
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

# Tuning configuration
GET /api/v1/config

# All users
GET /api/v1/users

# User profile
GET /api/v1/users/{user_id}/profile
```

## Response Format

```json
{
    "user_id": "41786230378",
    "user_info": {
        "user_id": "41786230378",
        "user_type": "loyal",
        "spending_segment": "average",
        "total_purchases": 1523,
        "unique_items": 487,
        "avg_item_price": 4.12,
        "last_purchase_date": "2025-01-10T14:30:00"
    },
    "recommendations": [
        {
            "item_id": "7003858505",
            "relevance_score": 0.87,
            "confidence_score": 0.92,
            "item_price": 4.99,
            "recommendation_reason": "Similar to 5 items in your purchase history",
            "model_source": "item_cf"
        }
    ],
    "primary_model": "hybrid",
    "fallback_used": false,
    "processing_time_ms": 45.2,
    "generated_at": "2025-01-15T10:30:00"
}
```

## Project Structure

```
Product-Recommendation-System/
├── backend/
│   ├── app/
│   │   ├── api/           # API routes
│   │   ├── core/          # Configuration & tuning knobs
│   │   ├── models/        # Recommendation models
│   │   │   ├── als_model.py
│   │   │   ├── item_cf_model.py
│   │   │   ├── popularity_model.py
│   │   │   ├── candidate_ranker.py
│   │   │   └── model_loader.py
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   └── utils/         # Data loading/cleaning
│   ├── tests/             # Test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── RecommendationCard.tsx
│   │   │   ├── RecommendationResults.tsx
│   │   │   ├── SimilarItems.tsx      # Similar items modal
│   │   │   ├── PopularItems.tsx
│   │   │   ├── UserSelector.tsx
│   │   │   └── ModelSelector.tsx
│   │   ├── hooks/         # Custom hooks
│   │   ├── api/           # API client
│   │   └── types/         # TypeScript types
│   └── package.json
├── scripts/
│   ├── train_models.py    # Offline model training
│   └── data_cleaning.py
├── models/                 # Serialized models (generated)
├── data/
│   └── Data Science - Assignment.xlsx
└── notebooks/
    └── data_analysis.ipynb
```

## Configuration

### Tuning Parameters

All tunable parameters are in `backend/app/core/tuning_config.py`:

```python
TUNING_CONFIG = {
    # Candidate generation
    "candidate_pool_size": 200,

    # Model weights for existing users
    "model_weights": {
        "als": 0.4,
        "item_cf": 0.4,
        "popularity": 0.2
    },

    # Upsell configuration
    "upsell": {
        "enabled": True,
        "factor": 0.1,           # 0 = no upsell, 1 = max upsell
        "price_boost_weight": 0.2
    },

    # User segment thresholds (percentiles)
    "user_segments": {
        "small": 0.25,
        "low_average": 0.50,
        "average": 0.75,
        "high": 1.0
    },

    # Repurchase cycle
    "repurchase_cycle": {
        "enabled": True,
        "default_cycle_days": 30
    },

    # Price sensitivity
    "price_sensitivity": {
        "budget_tolerance": 1.5   # Allow up to 1.5x user's avg price
    }
}
```

### Environment Variables

Create `.env` in backend/:

```env
DEBUG=false
ENVIRONMENT=development
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
CACHE_TTL_SECONDS=3600
MODEL_PATH=../models
```

## Frontend Features

- **Model Selection**: Choose between ALS, Item-CF, or Hybrid models
- **Similar Items Modal**: Click "Similar" button on any recommendation to see top 5 similar items
- **User Profiles**: View user spending segment and purchase history
- **Popular Items**: Click search icon to find similar items for trending products
- **Relevance & Confidence Scores**: Visual progress bars for each recommendation

## Running Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

## Performance

- Target latency: <100ms for recommendations
- Caching: TTL-based caching for repeated requests
- Precomputation: Models trained offline and loaded at startup
- Two-level architecture reduces computation at request time

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────────────────────────┐
│                 │     │           Backend (FastAPI)          │
│    Frontend     │     │                                      │
│    (React)      │────▶│  ┌─────────────────────────────┐    │
│                 │     │  │    Recommendation Service     │    │
└─────────────────┘     │  └─────────────────────────────┘    │
                        │              │                       │
                        │              ▼                       │
                        │  ┌─────────────────────────────┐    │
                        │  │     Level 1: Candidates      │    │
                        │  │  ┌─────┐ ┌─────┐ ┌───────┐  │    │
                        │  │  │ ALS │ │Item │ │Popular│  │    │
                        │  │  │     │ │ CF  │ │ ity   │  │    │
                        │  │  └─────┘ └─────┘ └───────┘  │    │
                        │  └─────────────────────────────┘    │
                        │              │                       │
                        │              ▼                       │
                        │  ┌─────────────────────────────┐    │
                        │  │   Level 2: Re-ranking        │    │
                        │  │  • User segment filtering    │    │
                        │  │  • Upsell boost              │    │
                        │  │  • Repurchase exclusion      │    │
                        │  │  • Price sensitivity         │    │
                        │  └─────────────────────────────┘    │
                        │              │                       │
                        │              ▼                       │
                        │  ┌─────────────────────────────┐    │
                        │  │    Serialized Models         │    │
                        │  │  (Loaded at startup)         │    │
                        │  └─────────────────────────────┘    │
                        └─────────────────────────────────────┘
```

## License

MIT License
