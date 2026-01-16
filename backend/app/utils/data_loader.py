"""Data loading utilities for the recommendation system"""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.utils.data_cleaner import DataCleaner


class DataLoader:
    """Load and prepare data for the recommendation system"""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or settings.excel_path
        self.cleaner = DataCleaner()
        self._loyal_df: Optional[pd.DataFrame] = None
        self._new_df: Optional[pd.DataFrame] = None
        self._combined_df: Optional[pd.DataFrame] = None
        self._item_catalog: Optional[pd.DataFrame] = None
        self._is_loaded = False

    def load_data(self) -> bool:
        """
        Load and clean all data from the Excel file.

        Returns:
            True if data was loaded successfully
        """
        try:
            logger.info(f"Loading data from {self.data_path}")

            if not self.data_path.exists():
                logger.error(f"Data file not found: {self.data_path}")
                return False

            # Load Excel file
            xlsx = pd.ExcelFile(self.data_path)
            logger.info(f"Found sheets: {xlsx.sheet_names}")

            # Load Loyal Customers
            if "Loyal Customers" in xlsx.sheet_names:
                loyal_raw = pd.read_excel(xlsx, sheet_name="Loyal Customers")
                self._loyal_df = self.cleaner.clean_transactions(loyal_raw, "loyal_customers")
                self._loyal_df["user_type"] = "loyal"
                logger.info(f"Loaded {len(self._loyal_df)} loyal customer transactions")

            # Load New Customers
            if "New Customers" in xlsx.sheet_names:
                new_raw = pd.read_excel(xlsx, sheet_name="New Customers")
                self._new_df = self.cleaner.clean_transactions(new_raw, "new_customers")
                self._new_df["user_type"] = "new"
                logger.info(f"Loaded {len(self._new_df)} new customer transactions")

            # Create combined dataset
            self._create_combined_dataset()

            # Build item catalog
            self._build_item_catalog()

            self._is_loaded = True
            logger.info("Data loading complete")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def _create_combined_dataset(self) -> None:
        """Create combined dataset from loyal and new customers"""
        dfs = []

        if self._loyal_df is not None and len(self._loyal_df) > 0:
            dfs.append(self._loyal_df)

        if self._new_df is not None and len(self._new_df) > 0:
            dfs.append(self._new_df)

        if dfs:
            self._combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined dataset: {len(self._combined_df)} transactions")

    def _build_item_catalog(self) -> None:
        """Build item catalog with aggregated statistics"""
        if self._combined_df is None:
            return

        # Aggregate item statistics
        self._item_catalog = self._combined_df.groupby("item_id").agg({
            "user_id": "nunique",
            "ticket_number": "count",
            "units_sold": "sum",
            "item_price": ["mean", "min", "max"]
        }).reset_index()

        # Flatten column names
        self._item_catalog.columns = [
            "item_id", "unique_buyers", "purchase_count",
            "total_units", "avg_price", "min_price", "max_price"
        ]

        # Calculate popularity score
        self._item_catalog["popularity_score"] = (
            self._item_catalog["purchase_count"] /
            self._item_catalog["purchase_count"].max()
        )

        logger.info(f"Built catalog with {len(self._item_catalog)} unique items")

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded"""
        return self._is_loaded

    @property
    def loyal_customers(self) -> Optional[pd.DataFrame]:
        """Get loyal customers data"""
        return self._loyal_df

    @property
    def new_customers(self) -> Optional[pd.DataFrame]:
        """Get new customers data"""
        return self._new_df

    @property
    def combined_data(self) -> Optional[pd.DataFrame]:
        """Get combined dataset"""
        return self._combined_df

    @property
    def item_catalog(self) -> Optional[pd.DataFrame]:
        """Get item catalog"""
        return self._item_catalog

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """
        Get purchase history for a specific user.

        Args:
            user_id: User identifier

        Returns:
            DataFrame with user's purchase history
        """
        if self._combined_df is None:
            return pd.DataFrame()

        return self._combined_df[
            self._combined_df["user_id"] == user_id
        ].sort_values("ticket_datetime", ascending=False)

    def get_user_type(self, user_id: str) -> str:
        """
        Determine user type based on their data presence.

        Args:
            user_id: User identifier

        Returns:
            'loyal', 'new', or 'unknown'
        """
        if self._loyal_df is not None and user_id in self._loyal_df["user_id"].values:
            return "loyal"
        if self._new_df is not None and user_id in self._new_df["user_id"].values:
            return "new"
        return "unknown"

    def get_item_info(self, item_id: str) -> Optional[dict]:
        """
        Get information about a specific item.

        Args:
            item_id: Item identifier

        Returns:
            Dictionary with item information or None
        """
        if self._item_catalog is None:
            return None

        item_data = self._item_catalog[self._item_catalog["item_id"] == item_id]
        if len(item_data) == 0:
            return None

        return item_data.iloc[0].to_dict()

    def get_popular_items(self, n: int = 10) -> list[str]:
        """
        Get top N popular items.

        Args:
            n: Number of items to return

        Returns:
            List of item IDs
        """
        if self._item_catalog is None:
            return []

        return self._item_catalog.nlargest(n, "purchase_count")["item_id"].tolist()

    def get_statistics(self) -> dict:
        """Get comprehensive data statistics"""
        stats = {
            "is_loaded": self._is_loaded,
            "cleaning_report": self.cleaner.get_cleaning_report(),
        }

        if self._loyal_df is not None:
            stats["loyal_customers"] = {
                "total_transactions": len(self._loyal_df),
                "unique_users": self._loyal_df["user_id"].nunique(),
                "unique_items": self._loyal_df["item_id"].nunique(),
                "date_range": {
                    "start": str(self._loyal_df["ticket_datetime"].min()),
                    "end": str(self._loyal_df["ticket_datetime"].max())
                }
            }

        if self._new_df is not None:
            stats["new_customers"] = {
                "total_transactions": len(self._new_df),
                "unique_users": self._new_df["user_id"].nunique(),
                "unique_items": self._new_df["item_id"].nunique(),
                "date_range": {
                    "start": str(self._new_df["ticket_datetime"].min()),
                    "end": str(self._new_df["ticket_datetime"].max())
                }
            }

        if self._item_catalog is not None:
            stats["item_catalog"] = {
                "total_items": len(self._item_catalog),
                "price_range": {
                    "min": float(self._item_catalog["min_price"].min()),
                    "max": float(self._item_catalog["max_price"].max()),
                    "avg": float(self._item_catalog["avg_price"].mean())
                }
            }

        return stats
