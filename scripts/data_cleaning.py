#!/usr/bin/env python3
"""
Data Cleaning Script for Product Recommendation System

This script cleans the raw data by:
1. Removing items with zero average price (promotional items, samples, etc.)
2. Validating data integrity
3. Saving cleaned data for model training

Usage:
    python scripts/data_cleaning.py

Output:
    - data/loyal_customers_cleaned.csv
    - data/new_customers_cleaned.csv
    - data/removed_items.csv (log of removed items)
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

    logger.info(f"Loaded {len(loyal_df):,} loyal customer records")
    logger.info(f"Loaded {len(new_df):,} new customer records")

    return loyal_df, new_df


def identify_zero_price_items(loyal_df: pd.DataFrame, new_df: pd.DataFrame) -> list:
    """Identify items with zero average price."""

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
    logger.info(f"Total unique zero price items: {all_zero_price_items}")

    return all_zero_price_items


def get_removal_stats(df: pd.DataFrame, items_to_remove: list, dataset_name: str) -> dict:
    """Get statistics about items to be removed."""
    stats = {}

    for item in items_to_remove:
        item_data = df[df['item_id'] == item]
        if len(item_data) > 0:
            stats[item] = {
                'dataset': dataset_name,
                'occurrences': len(item_data),
                'unique_users': item_data['user_id'].nunique(),
                'unique_transactions': item_data['ticket_number'].nunique(),
                'avg_price': item_data['item_price'].mean(),
                'total_units': item_data['units_sold'].sum()
            }
            logger.info(f"  Item '{item}' in {dataset_name}: {len(item_data)} occurrences, "
                       f"{item_data['user_id'].nunique()} users, avg_price=${item_data['item_price'].mean():.2f}")

    return stats


def remove_zero_price_items(df: pd.DataFrame, items_to_remove: list, dataset_name: str) -> pd.DataFrame:
    """Remove items with zero average price from dataset."""
    original_count = len(df)

    # Remove rows with zero price items
    cleaned_df = df[~df['item_id'].isin(items_to_remove)].copy()

    removed_count = original_count - len(cleaned_df)

    logger.info(f"{dataset_name}: Removed {removed_count:,} records ({removed_count/original_count*100:.2f}%)")
    logger.info(f"{dataset_name}: Cleaned dataset has {len(cleaned_df):,} records")

    return cleaned_df


def validate_cleaned_data(df: pd.DataFrame, dataset_name: str) -> bool:
    """Validate the cleaned dataset."""
    issues = []

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Check for negative prices
    neg_prices = (df['item_price'] < 0).sum()
    if neg_prices > 0:
        issues.append(f"{neg_prices} records with negative prices")

    # Check for zero prices remaining
    zero_avg_items = df.groupby('item_id')['item_price'].mean()
    zero_avg_items = zero_avg_items[zero_avg_items == 0]
    if len(zero_avg_items) > 0:
        issues.append(f"Zero average price items still present: {zero_avg_items.index.tolist()}")

    # Check for negative units
    neg_units = (df['units_sold'] < 0).sum()
    if neg_units > 0:
        issues.append(f"{neg_units} records with negative units_sold")

    if issues:
        logger.warning(f"{dataset_name} validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    logger.info(f"{dataset_name} validation passed!")
    return True


def save_cleaned_data(loyal_df: pd.DataFrame, new_df: pd.DataFrame,
                      removal_stats: dict, data_path: Path):
    """Save cleaned data and removal log."""

    # Save cleaned data
    loyal_output = data_path / 'loyal_customers_cleaned.csv'
    new_output = data_path / 'new_customers_cleaned.csv'

    loyal_df.to_csv(loyal_output, index=False)
    new_df.to_csv(new_output, index=False)

    logger.info(f"Saved cleaned loyal data to {loyal_output}")
    logger.info(f"Saved cleaned new data to {new_output}")

    # Save removal log
    if removal_stats:
        removal_log = pd.DataFrame.from_dict(removal_stats, orient='index')
        removal_log.index.name = 'item_id'
        removal_log = removal_log.reset_index()
        removal_log_path = data_path / 'removed_items.csv'
        removal_log.to_csv(removal_log_path, index=False)
        logger.info(f"Saved removal log to {removal_log_path}")

    # Save cleaning summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'loyal_records': len(loyal_df),
        'new_records': len(new_df),
        'loyal_users': loyal_df['user_id'].nunique(),
        'new_users': new_df['user_id'].nunique(),
        'loyal_items': loyal_df['item_id'].nunique(),
        'new_items': new_df['item_id'].nunique(),
        'items_removed': list(removal_stats.keys()) if removal_stats else []
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

    # Step 1: Load raw data
    logger.info("\nStep 1: Loading raw data...")
    loyal_df, new_df = load_raw_data(data_path)

    # Step 2: Identify zero price items
    logger.info("\nStep 2: Identifying zero price items...")
    zero_price_items = identify_zero_price_items(loyal_df, new_df)

    # Step 3: Get removal statistics
    logger.info("\nStep 3: Analyzing items to be removed...")
    removal_stats = {}
    removal_stats.update(get_removal_stats(loyal_df, zero_price_items, 'loyal'))
    removal_stats.update(get_removal_stats(new_df, zero_price_items, 'new'))

    # Step 4: Remove zero price items
    logger.info("\nStep 4: Removing zero price items...")
    loyal_cleaned = remove_zero_price_items(loyal_df, zero_price_items, 'Loyal')
    new_cleaned = remove_zero_price_items(new_df, zero_price_items, 'New')

    # Step 5: Validate cleaned data
    logger.info("\nStep 5: Validating cleaned data...")
    loyal_valid = validate_cleaned_data(loyal_cleaned, 'Loyal')
    new_valid = validate_cleaned_data(new_cleaned, 'New')

    if not (loyal_valid and new_valid):
        logger.error("Validation failed! Please review the issues above.")
        return 1

    # Step 6: Save cleaned data
    logger.info("\nStep 6: Saving cleaned data...")
    save_cleaned_data(loyal_cleaned, new_cleaned, removal_stats, data_path)

    logger.info("\n" + "=" * 60)
    logger.info("DATA CLEANING COMPLETE!")
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"\nItems removed: {zero_price_items}")
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
    print(f"  - {data_path / 'removed_items.csv'}")
    print(f"  - {data_path / 'cleaning_summary.csv'}")

    return 0


if __name__ == '__main__':
    exit(main())
