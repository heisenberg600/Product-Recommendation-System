#!/usr/bin/env python3
"""
Data Training Preparation Script for Product Recommendation System

This script prepares cleaned data for model training by:
1. Loading cleaned data (or cleaning if not available)
2. Computing user-item interaction matrices
3. Calculating item similarities
4. Creating user spending segments
5. Preparing train/test splits
6. Saving all artifacts for model training

Usage:
    python scripts/data_training.py

Prerequisites:
    Run data_cleaning.py first, or this script will run it automatically.

Output:
    - data/train/user_item_matrix.npz
    - data/train/item_similarities.csv
    - data/train/user_profiles.csv
    - data/train/item_features.csv
    - data/train/train_interactions.csv
    - data/train/test_interactions.csv
"""

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_cleaned_data(data_path: Path) -> bool:
    """Check if cleaned data exists, run cleaning if not."""
    loyal_cleaned = data_path / 'loyal_customers_cleaned.csv'
    new_cleaned = data_path / 'new_customers_cleaned.csv'

    if loyal_cleaned.exists() and new_cleaned.exists():
        logger.info("Cleaned data files found.")
        return True

    logger.warning("Cleaned data not found. Running data cleaning script...")
    cleaning_script = Path(__file__).parent / 'data_cleaning.py'

    if cleaning_script.exists():
        result = subprocess.run([sys.executable, str(cleaning_script),
                                '--data-path', str(data_path)],
                               capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Data cleaning failed: {result.stderr}")
            return False
        logger.info("Data cleaning completed successfully.")
        return True
    else:
        logger.error(f"Cleaning script not found: {cleaning_script}")
        return False


def load_cleaned_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned data files."""
    loyal_df = pd.read_csv(data_path / 'loyal_customers_cleaned.csv',
                           parse_dates=['ticket_datetime'])
    new_df = pd.read_csv(data_path / 'new_customers_cleaned.csv',
                         parse_dates=['ticket_datetime'])

    # Ensure item_id is string
    loyal_df['item_id'] = loyal_df['item_id'].astype(str)
    new_df['item_id'] = new_df['item_id'].astype(str)

    logger.info(f"Loaded {len(loyal_df):,} loyal customer records")
    logger.info(f"Loaded {len(new_df):,} new customer records")

    return loyal_df, new_df


def create_user_item_matrix(df: pd.DataFrame) -> tuple[sparse.csr_matrix, list, list]:
    """Create sparse user-item interaction matrix."""
    # Create mappings
    users = df['user_id'].unique().tolist()
    items = df['item_id'].unique().tolist()

    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {item: i for i, item in enumerate(items)}

    # Create interaction counts
    interactions = df.groupby(['user_id', 'item_id']).size().reset_index(name='count')

    rows = [user_to_idx[u] for u in interactions['user_id']]
    cols = [item_to_idx[i] for i in interactions['item_id']]
    data = interactions['count'].values

    matrix = sparse.csr_matrix((data, (rows, cols)),
                                shape=(len(users), len(items)))

    logger.info(f"Created user-item matrix: {matrix.shape}")
    logger.info(f"Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")

    return matrix, users, items


def compute_item_similarities(matrix: sparse.csr_matrix, items: list,
                              top_k: int = 50) -> pd.DataFrame:
    """Compute item-item similarities using cosine similarity."""
    logger.info("Computing item similarities...")

    # Transpose to get item vectors
    item_matrix = matrix.T.toarray()

    n_items = len(items)
    similarities = []

    for i in range(n_items):
        if i % 500 == 0:
            logger.info(f"  Processing item {i}/{n_items}")

        item_vec = item_matrix[i]
        if np.sum(item_vec) == 0:
            continue

        sims = []
        for j in range(n_items):
            if i != j:
                other_vec = item_matrix[j]
                if np.sum(other_vec) > 0:
                    sim = 1 - cosine(item_vec, other_vec)
                    if not np.isnan(sim) and sim > 0.01:
                        sims.append((j, sim))

        # Keep top_k similar items
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, sim in sims[:top_k]:
            similarities.append({
                'item_id': items[i],
                'similar_item_id': items[j],
                'similarity': sim
            })

    sim_df = pd.DataFrame(similarities)
    logger.info(f"Computed {len(sim_df)} item similarity pairs")

    return sim_df


def create_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Create user profile features for recommendations."""
    user_profiles = df.groupby('user_id').agg({
        'item_price': ['mean', 'median', 'std', 'min', 'max'],
        'item_id': 'nunique',
        'ticket_number': 'nunique',
        'units_sold': 'sum',
        'ticket_datetime': ['min', 'max']
    }).round(4)

    user_profiles.columns = ['avg_item_price', 'median_item_price', 'std_item_price',
                             'min_item_price', 'max_item_price',
                             'unique_items', 'num_transactions', 'total_units',
                             'first_purchase', 'last_purchase']

    user_profiles = user_profiles.reset_index()

    # Add spending segment
    p33 = user_profiles['avg_item_price'].quantile(0.33)
    p67 = user_profiles['avg_item_price'].quantile(0.67)

    user_profiles['spending_segment'] = user_profiles['avg_item_price'].apply(
        lambda x: 'Frugal' if x <= p33 else ('Big Spender' if x >= p67 else 'Normal')
    )

    # Calculate days since last purchase
    max_date = user_profiles['last_purchase'].max()
    user_profiles['days_since_last'] = (max_date - user_profiles['last_purchase']).dt.days

    logger.info(f"Created profiles for {len(user_profiles)} users")
    logger.info(f"Spending segments: {user_profiles['spending_segment'].value_counts().to_dict()}")

    return user_profiles


def create_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create item feature matrix for recommendations."""
    item_features = df.groupby('item_id').agg({
        'item_price': ['mean', 'std', 'min', 'max'],
        'user_id': 'nunique',
        'ticket_number': 'nunique',
        'units_sold': 'sum',
        'ticket_datetime': ['min', 'max']
    }).round(4)

    item_features.columns = ['avg_price', 'std_price', 'min_price', 'max_price',
                             'unique_buyers', 'num_transactions', 'total_units_sold',
                             'first_sold', 'last_sold']

    item_features = item_features.reset_index()

    # Add price segment
    p33 = item_features['avg_price'].quantile(0.33)
    p67 = item_features['avg_price'].quantile(0.67)

    item_features['price_segment'] = item_features['avg_price'].apply(
        lambda x: 'Budget' if x <= p33 else ('Premium' if x >= p67 else 'Mid-Range')
    )

    # Add popularity score (normalized)
    item_features['popularity_score'] = (
        item_features['total_units_sold'] / item_features['total_units_sold'].max()
    ).round(4)

    # Recency score (how recently was it sold)
    max_date = item_features['last_sold'].max()
    item_features['days_since_sold'] = (max_date - item_features['last_sold']).dt.days
    item_features['recency_score'] = (
        1 - item_features['days_since_sold'] / item_features['days_since_sold'].max()
    ).round(4)

    logger.info(f"Created features for {len(item_features)} items")
    logger.info(f"Price segments: {item_features['price_segment'].value_counts().to_dict()}")

    return item_features


def create_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                            time_based: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test split for model evaluation."""
    if time_based:
        # Time-based split: use recent transactions for testing
        df = df.sort_values('ticket_datetime')
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        logger.info("Using time-based split")
    else:
        # Random split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        logger.info("Using random split")

    logger.info(f"Train set: {len(train_df):,} records")
    logger.info(f"Test set: {len(test_df):,} records")

    return train_df, test_df


def save_training_artifacts(data_path: Path, matrix: sparse.csr_matrix,
                           users: list, items: list, item_similarities: pd.DataFrame,
                           user_profiles: pd.DataFrame, item_features: pd.DataFrame,
                           train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save all training artifacts."""
    train_path = data_path / 'train'
    train_path.mkdir(exist_ok=True)

    # Save user-item matrix
    sparse.save_npz(train_path / 'user_item_matrix.npz', matrix)

    # Save mappings
    mappings = {
        'users': users,
        'items': items,
        'user_to_idx': {str(u): i for i, u in enumerate(users)},
        'item_to_idx': {str(i): idx for idx, i in enumerate(items)}
    }
    with open(train_path / 'mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)

    # Save item similarities
    item_similarities.to_csv(train_path / 'item_similarities.csv', index=False)

    # Save user profiles
    user_profiles.to_csv(train_path / 'user_profiles.csv', index=False)

    # Save item features
    item_features.to_csv(train_path / 'item_features.csv', index=False)

    # Save train/test splits
    train_df.to_csv(train_path / 'train_interactions.csv', index=False)
    test_df.to_csv(train_path / 'test_interactions.csv', index=False)

    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'n_users': len(users),
        'n_items': len(items),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'matrix_shape': list(matrix.shape),
        'matrix_nnz': matrix.nnz,
        'sparsity': 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    }
    with open(train_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved all training artifacts to {train_path}")


def main():
    """Main function to run data training preparation."""
    parser = argparse.ArgumentParser(description='Prepare data for model training')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to data directory (default: data)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--time-split', action='store_true', default=True,
                       help='Use time-based split (default: True)')
    parser.add_argument('--top-k-similar', type=int, default=50,
                       help='Number of similar items to keep (default: 50)')
    args = parser.parse_args()

    data_path = Path(args.data_path)

    logger.info("=" * 60)
    logger.info("DATA TRAINING PREPARATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Ensure cleaned data exists
    logger.info("\nStep 1: Checking for cleaned data...")
    if not ensure_cleaned_data(data_path):
        logger.error("Cannot proceed without cleaned data.")
        return 1

    # Step 2: Load cleaned data
    logger.info("\nStep 2: Loading cleaned data...")
    loyal_df, new_df = load_cleaned_data(data_path)

    # Combine for training (focus on loyal customers who have more history)
    combined_df = pd.concat([loyal_df, new_df], ignore_index=True)
    combined_df = combined_df.sort_values('ticket_datetime')
    logger.info(f"Combined dataset: {len(combined_df):,} records")

    # Step 3: Create user-item matrix
    logger.info("\nStep 3: Creating user-item matrix...")
    matrix, users, items = create_user_item_matrix(combined_df)

    # Step 4: Compute item similarities
    logger.info("\nStep 4: Computing item similarities...")
    item_similarities = compute_item_similarities(matrix, items, top_k=args.top_k_similar)

    # Step 5: Create user profiles
    logger.info("\nStep 5: Creating user profiles...")
    user_profiles = create_user_profiles(combined_df)

    # Step 6: Create item features
    logger.info("\nStep 6: Creating item features...")
    item_features = create_item_features(combined_df)

    # Step 7: Create train/test split
    logger.info("\nStep 7: Creating train/test split...")
    train_df, test_df = create_train_test_split(combined_df, test_size=args.test_size,
                                                 time_based=args.time_split)

    # Step 8: Save all artifacts
    logger.info("\nStep 8: Saving training artifacts...")
    save_training_artifacts(data_path, matrix, users, items, item_similarities,
                           user_profiles, item_features, train_df, test_df)

    logger.info("\n" + "=" * 60)
    logger.info("DATA TRAINING PREPARATION COMPLETE!")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"\nMatrix dimensions: {matrix.shape[0]} users x {matrix.shape[1]} items")
    print(f"Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}")
    print(f"Total interactions: {matrix.nnz:,}")
    print(f"\nItem similarities computed: {len(item_similarities):,} pairs")
    print(f"User profiles created: {len(user_profiles)}")
    print(f"Item features created: {len(item_features)}")
    print(f"\nTrain set: {len(train_df):,} records")
    print(f"Test set: {len(test_df):,} records")
    print(f"\nOutput directory: {data_path / 'train'}")

    return 0


if __name__ == '__main__':
    exit(main())
