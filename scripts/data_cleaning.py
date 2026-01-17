#!/usr/bin/env python3
"""
Data Cleaning Script for Product Recommendation System

This script cleans the raw data by:
1. Standardizing column names
2. Converting data types
3. Validating data integrity
4. Saving cleaned data for model training

NOTE: Zero price items are KEPT (not removed) as they may be promotional items
that still provide valuable co-purchase signals.

Usage:
    python scripts/data_cleaning.py

Output:
    - data/loyal_customers_cleaned.csv
    - data/new_customers_cleaned.csv
    - data/zero_price_items.csv (log of zero price items for reference)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from Excel file."""
    excel_path = data_path / 'Data Science - Assignment.xlsx'

    if not excel_path.exists():
        raise FileNotFoundError(f"Data file not found: {excel_path}")

    logger.info(f"Loading data from {excel_path}")

    xlsx = pd.ExcelFile(excel_path)

    # Load both datasets
    loyal_df = pd.read_excel(xlsx, sheet_name='Loyal Customers')
    new_df = pd.read_excel(xlsx, sheet_name='New Customers')

    # Standardize column names
    loyal_df.columns = ['pos_number', 'ticket_number', 'ticket_datetime', 'ticket_amount',
                        'user_id', 'item_id', 'units_sold', 'item_price']
    new_df.columns = ['ticket_number', 'pos_number', 'ticket_datetime', 'ticket_amount',
                      'user_id', 'item_id', 'units_sold', 'item_price']

    # Convert item_id to string for consistency
    loyal_df['item_id'] = loyal_df['item_id'].astype(str)
    new_df['item_id'] = new_df['item_id'].astype(str)

    # Convert user_id to string for consistency
    loyal_df['user_id'] = loyal_df['user_id'].astype(str)
    new_df['user_id'] = new_df['user_id'].astype(str)

    logger.info(f"Loaded {len(loyal_df):,} loyal customer records")
    logger.info(f"Loaded {len(new_df):,} new customer records")

    return loyal_df, new_df


def identify_zero_price_items(loyal_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """
    Identify items with zero average price.
    These are logged but NOT removed (kept for co-purchase signals).
    """
    # Calculate average price per item
    loyal_item_avg_price = loyal_df.groupby('item_id')['item_price'].mean()
    new_item_avg_price = new_df.groupby('item_id')['item_price'].mean()

    # Find items with zero average price
    zero_price_items_loyal = loyal_item_avg_price[loyal_item_avg_price == 0].index.tolist()
    zero_price_items_new = new_item_avg_price[new_item_avg_price == 0].index.tolist()

    # Combine all zero price items
    all_zero_price_items = list(set(zero_price_items_loyal + zero_price_items_new))

    logger.info(f"Found {len(zero_price_items_loyal)} zero price items in loyal data: {zero_price_items_loyal}")
    logger.info(f"Found {len(zero_price_items_new)} zero price items in new data: {zero_price_items_new}")
    logger.info(f"Total unique zero price items (KEPT, not removed): {all_zero_price_items}")

    # Get stats for zero price items
    zero_price_stats = {}
    for item in all_zero_price_items:
        loyal_data = loyal_df[loyal_df['item_id'] == item]
        new_data = new_df[new_df['item_id'] == item]

        zero_price_stats[item] = {
            'loyal_occurrences': len(loyal_data),
            'new_occurrences': len(new_data),
            'total_occurrences': len(loyal_data) + len(new_data),
            'loyal_unique_users': loyal_data['user_id'].nunique() if len(loyal_data) > 0 else 0,
            'new_unique_users': new_data['user_id'].nunique() if len(new_data) > 0 else 0,
            'avg_price': 0.0
        }

    return zero_price_stats


def clean_data(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Clean the dataset:
    - Handle missing values
    - Fix data types
    - Remove duplicates if any
    """
    original_count = len(df)
    cleaned_df = df.copy()

    # Handle missing values
    null_before = cleaned_df.isnull().sum().sum()
    if null_before > 0:
        logger.warning(f"{dataset_name}: Found {null_before} null values")
        # Fill numeric nulls with 0, keep for investigation
        cleaned_df['item_price'] = cleaned_df['item_price'].fillna(0)
        cleaned_df['units_sold'] = cleaned_df['units_sold'].fillna(1)

    # Ensure datetime is properly parsed
    cleaned_df['ticket_datetime'] = pd.to_datetime(cleaned_df['ticket_datetime'])

    # Ensure numeric types
    cleaned_df['item_price'] = pd.to_numeric(cleaned_df['item_price'], errors='coerce').fillna(0)
    cleaned_df['units_sold'] = pd.to_numeric(cleaned_df['units_sold'], errors='coerce').fillna(1).astype(int)
    cleaned_df['ticket_amount'] = pd.to_numeric(cleaned_df['ticket_amount'], errors='coerce').fillna(0)

    # Remove exact duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    duplicates_removed = original_count - len(cleaned_df)
    if duplicates_removed > 0:
        logger.info(f"{dataset_name}: Removed {duplicates_removed} duplicate records")

    logger.info(f"{dataset_name}: Cleaned dataset has {len(cleaned_df):,} records")

    return cleaned_df


def validate_data(df: pd.DataFrame, dataset_name: str) -> bool:
    """Validate the dataset."""
    issues = []

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Check for negative prices
    neg_prices = (df['item_price'] < 0).sum()
    if neg_prices > 0:
        issues.append(f"{neg_prices} records with negative prices")

    # Check for negative units
    neg_units = (df['units_sold'] < 0).sum()
    if neg_units > 0:
        issues.append(f"{neg_units} records with negative units_sold")

    # Check date range
    min_date = df['ticket_datetime'].min()
    max_date = df['ticket_datetime'].max()
    logger.info(f"{dataset_name}: Date range {min_date} to {max_date}")

    if issues:
        logger.warning(f"{dataset_name} validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    logger.info(f"{dataset_name} validation passed!")
    return True


def save_cleaned_data(loyal_df: pd.DataFrame, new_df: pd.DataFrame,
                      zero_price_stats: dict, data_path: Path):
    """Save cleaned data and metadata."""

    # Save cleaned data
    loyal_output = data_path / 'loyal_customers_cleaned.csv'
    new_output = data_path / 'new_customers_cleaned.csv'

    loyal_df.to_csv(loyal_output, index=False)
    new_df.to_csv(new_output, index=False)

    logger.info(f"Saved cleaned loyal data to {loyal_output}")
    logger.info(f"Saved cleaned new data to {new_output}")

    # Save zero price items log (for reference, not removed)
    if zero_price_stats:
        zero_price_log = pd.DataFrame.from_dict(zero_price_stats, orient='index')
        zero_price_log.index.name = 'item_id'
        zero_price_log = zero_price_log.reset_index()
        zero_price_log_path = data_path / 'zero_price_items.csv'
        zero_price_log.to_csv(zero_price_log_path, index=False)
        logger.info(f"Saved zero price items log to {zero_price_log_path}")

    # Save cleaning summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'loyal_records': len(loyal_df),
        'new_records': len(new_df),
        'loyal_users': loyal_df['user_id'].nunique(),
        'new_users': new_df['user_id'].nunique(),
        'loyal_items': loyal_df['item_id'].nunique(),
        'new_items': new_df['item_id'].nunique(),
        'zero_price_items_count': len(zero_price_stats),
        'zero_price_items': list(zero_price_stats.keys()) if zero_price_stats else [],
        'note': 'Zero price items are KEPT (not removed)'
    }

    summary_df = pd.DataFrame([summary])
    summary_path = data_path / 'cleaning_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved cleaning summary to {summary_path}")


def main():
    """Main function to run data cleaning pipeline."""
    parser = argparse.ArgumentParser(description='Clean product recommendation data')
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to data directory (default: data)')
    args = parser.parse_args()

    data_path = Path(args.data_path)

    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    logger.info("NOTE: Zero price items are KEPT (not removed)")

    # Step 1: Load raw data
    logger.info("\nStep 1: Loading raw data...")
    loyal_df, new_df = load_raw_data(data_path)

    # Step 2: Identify zero price items (for logging, not removal)
    logger.info("\nStep 2: Identifying zero price items (will be KEPT)...")
    zero_price_stats = identify_zero_price_items(loyal_df, new_df)

    # Step 3: Clean data
    logger.info("\nStep 3: Cleaning data...")
    loyal_cleaned = clean_data(loyal_df, 'Loyal')
    new_cleaned = clean_data(new_df, 'New')

    # Step 4: Validate data
    logger.info("\nStep 4: Validating data...")
    loyal_valid = validate_data(loyal_cleaned, 'Loyal')
    new_valid = validate_data(new_cleaned, 'New')

    if not (loyal_valid and new_valid):
        logger.warning("Validation had issues. Data saved anyway for inspection.")

    # Step 5: Save cleaned data
    logger.info("\nStep 5: Saving cleaned data...")
    save_cleaned_data(loyal_cleaned, new_cleaned, zero_price_stats, data_path)

    logger.info("\n" + "=" * 60)
    logger.info("DATA CLEANING COMPLETE!")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"\nZero price items (KEPT): {list(zero_price_stats.keys())}")
    print(f"\nLoyal Customers:")
    print(f"  - Records: {len(loyal_cleaned):,}")
    print(f"  - Users: {loyal_cleaned['user_id'].nunique()}")
    print(f"  - Items: {loyal_cleaned['item_id'].nunique()}")
    print(f"\nNew Customers:")
    print(f"  - Records: {len(new_cleaned):,}")
    print(f"  - Users: {new_cleaned['user_id'].nunique()}")
    print(f"  - Items: {new_cleaned['item_id'].nunique()}")
    print(f"\nOutput files:")
    print(f"  - {data_path / 'loyal_customers_cleaned.csv'}")
    print(f"  - {data_path / 'new_customers_cleaned.csv'}")
    print(f"  - {data_path / 'zero_price_items.csv'}")
    print(f"  - {data_path / 'cleaning_summary.csv'}")

    return 0


if __name__ == '__main__':
    exit(main())
