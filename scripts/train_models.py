#!/usr/bin/env python3
"""
Model Training Script for Product Recommendation System

This script trains all recommendation models offline and serializes them to disk.
The backend loads these pre-trained models on startup for fast inference.

Models trained:
1. ALS (Matrix Factorization) - with spending/quantity ratio implicit feedback
2. Item-Item Collaborative Filtering - with weighted co-purchase similarity
3. Popularity Model - weighted by purchase count, unique buyers, quantity

Also computes:
- User spending segments (small, low_average, average, high)
- Item repurchase cycles
- User profiles
- Item features

Usage:
    python scripts/train_models.py --data-path data --output-path models

Output:
    models/
    ├── als_model.pkl
    ├── item_cf_model.pkl
    ├── popularity_model.pkl
    ├── user_profiles.pkl
    ├── item_features.pkl
    ├── repurchase_cycles.pkl
    └── metadata.json
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import argparse
import logging
import json
import pickle
from datetime import datetime
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.app.core.tuning_config import TUNING_CONFIG
except ImportError:
    # Fallback config if backend not available
    TUNING_CONFIG = {
        "user_segments": {"small": 0.25, "low_average": 0.50, "average": 0.75, "high": 1.0},
        "implicit_feedback": {"spending_ratio_weight": 0.5, "quantity_ratio_weight": 0.5,
                              "min_signal": 0.1, "max_signal": 5.0, "use_log_transform": True},
        "item_cf": {"similarity_top_k": 100, "min_co_purchase_count": 2,
                    "similarity_threshold": 0.01, "use_weighted_similarity": True},
        "als": {"n_factors": 64, "n_iterations": 15, "regularization": 0.01, "alpha": 40},
        "popularity_weights": {"purchase_count": 0.4, "unique_buyers": 0.4, "total_quantity": 0.2},
        "repurchase_cycle": {"min_purchases_for_cycle": 2, "default_cycle_days": 30}
    }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and serializes all recommendation models."""

    def __init__(self, config: dict = None):
        self.config = config or TUNING_CONFIG
        self.data = None
        self.user_avg_price = None
        self.item_avg_quantity = None
        self.user_to_idx = None
        self.idx_to_user = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.loyal_users = set()
        self.new_users = set()

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load loyal customers data for training."""
        loyal_path = data_path / 'loyal_customers_cleaned.csv'
        new_path = data_path / 'new_customers_cleaned.csv'

        if not loyal_path.exists():
            raise FileNotFoundError(
                f"Cleaned data not found. Run data_cleaning.py first.\n"
                f"Expected: {loyal_path}"
            )

        logger.info("Loading loyal customers data for training...")
        loyal_df = pd.read_csv(loyal_path, parse_dates=['ticket_datetime'])

        # Ensure string types
        loyal_df['item_id'] = loyal_df['item_id'].astype(str)
        loyal_df['user_id'] = loyal_df['user_id'].astype(str)

        # Track loyal users
        self.loyal_users = set(loyal_df['user_id'].unique())

        # Load new customers just to track them (not for training)
        if new_path.exists():
            new_df = pd.read_csv(new_path, parse_dates=['ticket_datetime'])
            new_df['user_id'] = new_df['user_id'].astype(str)
            self.new_users = set(new_df['user_id'].unique())
            logger.info(f"New users (tracked, not trained): {len(self.new_users):,}")
        else:
            self.new_users = set()

        logger.info(f"Loyal users (for training): {len(self.loyal_users):,}")

        # Use only loyal customers for training
        loyal_df = loyal_df.sort_values('ticket_datetime')

        logger.info(f"Total records: {len(loyal_df):,}")
        logger.info(f"Unique users: {loyal_df['user_id'].nunique():,}")
        logger.info(f"Unique items: {loyal_df['item_id'].nunique():,}")

        self.data = loyal_df
        return loyal_df

    def compute_implicit_feedback(self) -> pd.DataFrame:
        """
        Compute spending and quantity ratio signals for implicit feedback.

        spending_ratio = item_price / user_avg_item_price
        quantity_ratio = units_sold / item_avg_quantity
        """
        logger.info("Computing implicit feedback signals...")

        df = self.data.copy()
        config = self.config['implicit_feedback']

        # User average item price
        self.user_avg_price = df.groupby('user_id')['item_price'].mean().to_dict()

        # Item average quantity
        self.item_avg_quantity = df.groupby('item_id')['units_sold'].mean().to_dict()

        # Compute spending ratio
        def get_spending_ratio(row):
            user_avg = self.user_avg_price.get(row['user_id'], 1.0)
            if user_avg <= 0:
                return 1.0
            return row['item_price'] / user_avg

        # Compute quantity ratio
        def get_quantity_ratio(row):
            item_avg = self.item_avg_quantity.get(row['item_id'], 1.0)
            if item_avg <= 0:
                return 1.0
            return row['units_sold'] / item_avg

        df['spending_ratio'] = df.apply(get_spending_ratio, axis=1)
        df['quantity_ratio'] = df.apply(get_quantity_ratio, axis=1)

        # Apply log transform if configured
        if config.get('use_log_transform', True):
            df['spending_ratio'] = np.log1p(df['spending_ratio'])
            df['quantity_ratio'] = np.log1p(df['quantity_ratio'])

        # Clip to min/max
        min_signal = config.get('min_signal', 0.1)
        max_signal = config.get('max_signal', 5.0)
        df['spending_ratio'] = df['spending_ratio'].clip(min_signal, max_signal)
        df['quantity_ratio'] = df['quantity_ratio'].clip(min_signal, max_signal)

        # Combined implicit signal (weighted geometric mean)
        spending_weight = config.get('spending_ratio_weight', 0.5)
        quantity_weight = config.get('quantity_ratio_weight', 0.5)

        df['implicit_signal'] = (
            (df['spending_ratio'] ** spending_weight) *
            (df['quantity_ratio'] ** quantity_weight)
        )

        logger.info(f"Implicit signal stats - mean: {df['implicit_signal'].mean():.3f}, "
                   f"std: {df['implicit_signal'].std():.3f}")

        self.data = df
        return df

    def create_mappings(self):
        """Create user and item index mappings."""
        users = self.data['user_id'].unique().tolist()
        items = self.data['item_id'].unique().tolist()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {i: idx for idx, i in enumerate(items)}
        self.idx_to_item = {idx: i for i, idx in self.item_to_idx.items()}

        logger.info(f"Created mappings: {len(users)} users, {len(items)} items")

    def train_als_model(self) -> dict:
        """
        Train ALS matrix factorization model with implicit feedback.
        Uses the implicit library for efficient training.
        """
        logger.info("Training ALS model...")
        config = self.config['als']

        # Create user-item matrix with implicit signals
        interactions = self.data.groupby(['user_id', 'item_id'])['implicit_signal'].sum().reset_index()

        rows = [self.user_to_idx[u] for u in interactions['user_id']]
        cols = [self.item_to_idx[i] for i in interactions['item_id']]
        data = interactions['implicit_signal'].values

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        # Create sparse matrix
        user_item_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        )

        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        logger.info(f"Non-zero entries: {user_item_matrix.nnz:,}")

        # Train ALS using alternating least squares
        n_factors = config.get('n_factors', 64)
        n_iterations = config.get('n_iterations', 15)
        regularization = config.get('regularization', 0.01)
        alpha = config.get('alpha', 40)

        # Initialize factors randomly
        np.random.seed(42)
        user_factors = np.random.normal(0, 0.01, (n_users, n_factors))
        item_factors = np.random.normal(0, 0.01, (n_items, n_factors))

        # Confidence matrix: C = 1 + alpha * R
        confidence = user_item_matrix.copy()
        confidence.data = 1 + alpha * confidence.data

        # ALS iterations
        for iteration in range(n_iterations):
            # Update user factors
            for u in range(n_users):
                # Get items this user has interacted with
                user_items = user_item_matrix[u].indices
                if len(user_items) == 0:
                    continue

                # C_u * Y
                Cu = np.diag([confidence[u, i] for i in user_items])
                Yu = item_factors[user_items]

                # A = Y^T * C_u * Y + lambda * I
                A = Yu.T @ Cu @ Yu + regularization * np.eye(n_factors)
                # b = Y^T * C_u * p_u (p_u is binary preference)
                pu = np.ones(len(user_items))  # Implicit feedback = preference
                b = Yu.T @ Cu @ pu

                user_factors[u] = np.linalg.solve(A, b)

            # Update item factors
            for i in range(n_items):
                # Get users who have interacted with this item
                item_users = user_item_matrix[:, i].nonzero()[0]
                if len(item_users) == 0:
                    continue

                # C_i * X
                Ci = np.diag([confidence[u, i] for u in item_users])
                Xi = user_factors[item_users]

                # A = X^T * C_i * X + lambda * I
                A = Xi.T @ Ci @ Xi + regularization * np.eye(n_factors)
                # b = X^T * C_i * p_i
                pi = np.ones(len(item_users))
                b = Xi.T @ Ci @ pi

                item_factors[i] = np.linalg.solve(A, b)

            if (iteration + 1) % 5 == 0:
                logger.info(f"  ALS iteration {iteration + 1}/{n_iterations}")

        # Compute confidence scores for recommendations
        # Based on how many interactions the user has
        user_interaction_counts = np.array(user_item_matrix.sum(axis=1)).flatten()
        max_interactions = user_interaction_counts.max()
        user_confidence = np.minimum(user_interaction_counts / 50, 1.0)

        als_model = {
            'user_factors': user_factors,
            'item_factors': item_factors,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'user_confidence': user_confidence,
            'n_factors': n_factors,
            'config': config
        }

        logger.info("ALS model training complete")
        return als_model

    def train_item_cf_model(self) -> dict:
        """
        Train Item-Item Collaborative Filtering model.
        Uses weighted co-purchase similarity based on implicit signals.
        """
        logger.info("Training Item-CF model...")
        config = self.config['item_cf']

        # Build co-purchase matrix weighted by implicit signal
        # For each transaction, items bought together get similarity boost
        transaction_items = self.data.groupby('ticket_number').agg({
            'item_id': list,
            'implicit_signal': list
        }).reset_index()

        # Co-purchase counts with signal weighting
        item_copurchase = defaultdict(lambda: defaultdict(float))
        item_purchase_count = defaultdict(float)

        for _, row in transaction_items.iterrows():
            items = row['item_id']
            signals = row['implicit_signal']

            if len(items) < 2:
                continue

            for i, (item1, sig1) in enumerate(zip(items, signals)):
                item_purchase_count[item1] += sig1
                for j, (item2, sig2) in enumerate(zip(items, signals)):
                    if i != j:
                        # Weight by geometric mean of signals
                        weight = np.sqrt(sig1 * sig2)
                        item_copurchase[item1][item2] += weight

        # Compute cosine similarity
        min_copurchase = config.get('min_co_purchase_count', 2)
        similarity_threshold = config.get('similarity_threshold', 0.01)
        top_k = config.get('similarity_top_k', 100)

        item_similarities = {}
        all_items = list(self.item_to_idx.keys())

        for i, item in enumerate(all_items):
            if i % 1000 == 0:
                logger.info(f"  Processing item {i}/{len(all_items)}")

            if item not in item_copurchase:
                continue

            similarities = []
            item_count = item_purchase_count.get(item, 1)

            for other_item, copurchase_weight in item_copurchase[item].items():
                if copurchase_weight < min_copurchase:
                    continue

                other_count = item_purchase_count.get(other_item, 1)

                # Cosine similarity with weighting
                similarity = copurchase_weight / (np.sqrt(item_count) * np.sqrt(other_count))

                if similarity >= similarity_threshold:
                    similarities.append((other_item, float(similarity)))

            # Keep top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            item_similarities[item] = similarities[:top_k]

        # Compute confidence based on number of similar items found
        item_confidence = {}
        for item in all_items:
            n_similar = len(item_similarities.get(item, []))
            item_confidence[item] = min(n_similar / 20, 1.0)

        item_cf_model = {
            'similarities': item_similarities,
            'item_confidence': item_confidence,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'config': config
        }

        logger.info(f"Item-CF model: {len(item_similarities)} items with similarities")
        return item_cf_model

    def train_popularity_model(self) -> dict:
        """
        Train popularity model with weighted scoring.
        Score = w1*purchase_count + w2*unique_buyers + w3*total_quantity
        """
        logger.info("Training Popularity model...")
        weights = self.config['popularity_weights']

        # Compute popularity metrics per item
        item_stats = self.data.groupby('item_id').agg({
            'ticket_number': 'count',      # Purchase count
            'user_id': 'nunique',          # Unique buyers
            'units_sold': 'sum',           # Total quantity
            'item_price': 'mean',          # Average price
            'ticket_datetime': 'max'       # Last purchase date
        }).reset_index()

        item_stats.columns = ['item_id', 'purchase_count', 'unique_buyers',
                              'total_quantity', 'avg_price', 'last_purchase']

        # Normalize each metric to 0-1 range
        scaler = MinMaxScaler()
        for col in ['purchase_count', 'unique_buyers', 'total_quantity']:
            item_stats[f'{col}_norm'] = scaler.fit_transform(item_stats[[col]])

        # Compute weighted popularity score
        item_stats['popularity_score'] = (
            weights['purchase_count'] * item_stats['purchase_count_norm'] +
            weights['unique_buyers'] * item_stats['unique_buyers_norm'] +
            weights['total_quantity'] * item_stats['total_quantity_norm']
        )

        # Normalize final score
        item_stats['popularity_score'] = (
            item_stats['popularity_score'] / item_stats['popularity_score'].max()
        )

        # Compute recency score
        max_date = item_stats['last_purchase'].max()
        item_stats['days_since_purchase'] = (max_date - item_stats['last_purchase']).dt.days
        max_days = item_stats['days_since_purchase'].max()
        if max_days > 0:
            item_stats['recency_score'] = 1 - (item_stats['days_since_purchase'] / max_days)
        else:
            item_stats['recency_score'] = 1.0

        # Convert to dict for fast lookup
        popularity_scores = item_stats.set_index('item_id')[
            ['popularity_score', 'avg_price', 'recency_score', 'purchase_count', 'unique_buyers']
        ].to_dict('index')

        # Sort items by popularity for quick retrieval
        sorted_items = item_stats.sort_values('popularity_score', ascending=False)['item_id'].tolist()

        popularity_model = {
            'scores': popularity_scores,
            'sorted_items': sorted_items,
            'weights': weights
        }

        logger.info(f"Popularity model: {len(popularity_scores)} items scored")
        return popularity_model

    def compute_user_segments(self) -> dict:
        """
        Segment users by spending: small, low_average, average, high.
        Based on quartiles of average item price.
        """
        logger.info("Computing user segments...")
        thresholds = self.config['user_segments']

        # Compute user average item price
        user_avg = self.data.groupby('user_id')['item_price'].mean()

        # Compute percentile thresholds
        percentiles = user_avg.quantile([
            thresholds['small'],       # 25th percentile
            thresholds['low_average'], # 50th percentile
            thresholds['average']      # 75th percentile
        ])

        p25 = percentiles.iloc[0]
        p50 = percentiles.iloc[1]
        p75 = percentiles.iloc[2]

        def get_segment(avg_price):
            if avg_price <= p25:
                return 'small'
            elif avg_price <= p50:
                return 'low_average'
            elif avg_price <= p75:
                return 'average'
            else:
                return 'high'

        # Assign segments
        user_segments = {uid: get_segment(avg) for uid, avg in user_avg.items()}

        # Compute segment statistics
        segment_stats = {}
        for segment in ['small', 'low_average', 'average', 'high']:
            segment_users = [u for u, s in user_segments.items() if s == segment]
            segment_stats[segment] = {
                'count': len(segment_users),
                'avg_price_range': (
                    0 if segment == 'small' else
                    p25 if segment == 'low_average' else
                    p50 if segment == 'average' else p75,
                    p25 if segment == 'small' else
                    p50 if segment == 'low_average' else
                    p75 if segment == 'average' else user_avg.max()
                )
            }

        logger.info(f"User segments: {segment_stats}")

        return {
            'segments': user_segments,
            'thresholds': {'p25': p25, 'p50': p50, 'p75': p75},
            'stats': segment_stats
        }

    def compute_user_profiles(self, user_segments: dict, data_path: Path = None) -> dict:
        """Compute comprehensive user profiles for both loyal and new users."""
        logger.info("Computing user profiles...")

        user_profiles = {}

        # Process loyal users from training data
        user_history = self.data.groupby('user_id')

        for user_id, group in user_history:
            # Get last purchase per item for repurchase cycle checking
            item_last_purchase = group.groupby('item_id')['ticket_datetime'].max().to_dict()

            user_profiles[user_id] = {
                'is_loyal': True,
                'avg_item_price': float(group['item_price'].mean()),
                'median_item_price': float(group['item_price'].median()),
                'total_purchases': len(group),
                'unique_items': group['item_id'].nunique(),
                'segment': user_segments['segments'].get(user_id, 'average'),
                'first_purchase': group['ticket_datetime'].min().isoformat(),
                'last_purchase': group['ticket_datetime'].max().isoformat(),
                'item_last_purchase': {k: v.isoformat() for k, v in item_last_purchase.items()}
            }

        logger.info(f"Computed profiles for {len(user_profiles)} loyal users")

        # Also load and create profiles for new users
        if data_path:
            new_path = data_path / 'new_customers_cleaned.csv'
            if new_path.exists():
                logger.info("Loading new customers for profiles...")
                new_df = pd.read_csv(new_path, parse_dates=['ticket_datetime'])
                new_df['user_id'] = new_df['user_id'].astype(str)
                new_df['item_id'] = new_df['item_id'].astype(str)

                new_user_history = new_df.groupby('user_id')

                for user_id, group in new_user_history:
                    if user_id not in user_profiles:  # Don't overwrite loyal users
                        item_last_purchase = group.groupby('item_id')['ticket_datetime'].max().to_dict()

                        user_profiles[user_id] = {
                            'is_loyal': False,
                            'avg_item_price': float(group['item_price'].mean()),
                            'median_item_price': float(group['item_price'].median()),
                            'total_purchases': len(group),
                            'unique_items': group['item_id'].nunique(),
                            'segment': 'average',  # Default segment for new users
                            'first_purchase': group['ticket_datetime'].min().isoformat(),
                            'last_purchase': group['ticket_datetime'].max().isoformat(),
                            'item_last_purchase': {k: v.isoformat() for k, v in item_last_purchase.items()}
                        }

                new_count = sum(1 for p in user_profiles.values() if not p['is_loyal'])
                logger.info(f"Added profiles for {new_count} new users")

        logger.info(f"Total user profiles: {len(user_profiles)}")
        return user_profiles

    def compute_item_features(self, popularity_model: dict) -> dict:
        """Compute comprehensive item features."""
        logger.info("Computing item features...")

        item_features = {}

        for item_id, scores in popularity_model['scores'].items():
            item_features[item_id] = {
                'avg_price': float(scores['avg_price']),
                'popularity_score': float(scores['popularity_score']),
                'recency_score': float(scores['recency_score']),
                'purchase_count': int(scores['purchase_count']),
                'unique_buyers': int(scores['unique_buyers'])
            }

        logger.info(f"Computed features for {len(item_features)} items")
        return item_features

    def compute_repurchase_cycles(self) -> dict:
        """
        Compute average days between repurchases per item.
        Used to avoid recommending items bought too recently.
        """
        logger.info("Computing repurchase cycles...")
        min_purchases = self.config['repurchase_cycle']['min_purchases_for_cycle']
        default_cycle = self.config['repurchase_cycle']['default_cycle_days']

        cycles = {}

        # Group by user and item to find repurchase patterns
        for item_id in self.data['item_id'].unique():
            item_data = self.data[self.data['item_id'] == item_id]

            user_cycles = []
            for user_id in item_data['user_id'].unique():
                user_item_data = item_data[item_data['user_id'] == user_id].sort_values('ticket_datetime')

                if len(user_item_data) >= min_purchases:
                    dates = user_item_data['ticket_datetime']
                    diffs = dates.diff().dropna().dt.days
                    if len(diffs) > 0:
                        user_cycles.extend(diffs.tolist())

            if user_cycles:
                cycles[item_id] = {
                    'avg_cycle_days': float(np.mean(user_cycles)),
                    'median_cycle_days': float(np.median(user_cycles)),
                    'min_cycle_days': float(np.min(user_cycles)),
                    'max_cycle_days': float(np.max(user_cycles)),
                    'sample_size': len(user_cycles)
                }
            else:
                cycles[item_id] = {
                    'avg_cycle_days': float(default_cycle),
                    'median_cycle_days': float(default_cycle),
                    'min_cycle_days': float(default_cycle),
                    'max_cycle_days': float(default_cycle),
                    'sample_size': 0
                }

        logger.info(f"Computed repurchase cycles for {len(cycles)} items")
        return cycles

    def save_models(self, output_path: Path, models: dict):
        """Save all models and artifacts to disk."""
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving models to {output_path}...")

        # Save each model
        for name, model in models.items():
            if name == 'metadata':
                # Save metadata as JSON
                with open(output_path / 'metadata.json', 'w') as f:
                    json.dump(model, f, indent=2, default=str)
            else:
                # Save model as pickle
                with open(output_path / f'{name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"  Saved {name}.pkl")

        logger.info("All models saved successfully!")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train recommendation models')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--output-path', type=str, default='models',
                       help='Path to output models directory')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    logger.info("=" * 60)
    logger.info("MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    trainer = ModelTrainer(TUNING_CONFIG)

    # Step 1: Load data
    logger.info("\n" + "=" * 40)
    logger.info("Step 1: Loading data")
    logger.info("=" * 40)
    trainer.load_data(data_path)

    # Step 2: Compute implicit feedback
    logger.info("\n" + "=" * 40)
    logger.info("Step 2: Computing implicit feedback signals")
    logger.info("=" * 40)
    trainer.compute_implicit_feedback()

    # Step 3: Create mappings
    logger.info("\n" + "=" * 40)
    logger.info("Step 3: Creating user/item mappings")
    logger.info("=" * 40)
    trainer.create_mappings()

    # Step 4: Train models
    logger.info("\n" + "=" * 40)
    logger.info("Step 4: Training models")
    logger.info("=" * 40)

    als_model = trainer.train_als_model()
    item_cf_model = trainer.train_item_cf_model()
    popularity_model = trainer.train_popularity_model()

    # Step 5: Compute segments and profiles
    logger.info("\n" + "=" * 40)
    logger.info("Step 5: Computing user segments and profiles")
    logger.info("=" * 40)

    user_segments = trainer.compute_user_segments()
    user_profiles = trainer.compute_user_profiles(user_segments, data_path)
    item_features = trainer.compute_item_features(popularity_model)
    repurchase_cycles = trainer.compute_repurchase_cycles()

    # Step 6: Create metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'version': '1.0.0',
        'n_users': len(trainer.user_to_idx),
        'n_items': len(trainer.item_to_idx),
        'config': TUNING_CONFIG,
        'data_date_range': {
            'start': trainer.data['ticket_datetime'].min().isoformat(),
            'end': trainer.data['ticket_datetime'].max().isoformat()
        }
    }

    # Step 7: Save all models
    logger.info("\n" + "=" * 40)
    logger.info("Step 6: Saving models")
    logger.info("=" * 40)

    models = {
        'als_model': als_model,
        'item_cf_model': item_cf_model,
        'popularity_model': popularity_model,
        'user_segments': user_segments,
        'user_profiles': user_profiles,
        'item_features': item_features,
        'repurchase_cycles': repurchase_cycles,
        'metadata': metadata
    }

    trainer.save_models(output_path, models)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved to: {output_path}")
    print(f"\nFiles created:")
    for name in models:
        ext = '.json' if name == 'metadata' else '.pkl'
        print(f"  - {name}{ext}")
    print(f"\nTotal users: {len(trainer.user_to_idx):,}")
    print(f"Total items: {len(trainer.item_to_idx):,}")

    return 0


if __name__ == '__main__':
    exit(main())
