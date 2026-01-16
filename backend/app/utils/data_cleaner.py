"""Data cleaning utilities for the recommendation system"""

from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class DataCleaner:
    """Comprehensive data cleaning for transaction data"""

    def __init__(self):
        self.cleaning_stats: dict = {}

    def clean_transactions(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset"
    ) -> pd.DataFrame:
        """
        Clean transaction data with multiple cleaning steps.

        Args:
            df: Raw transaction DataFrame
            dataset_name: Name for logging purposes

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning for {dataset_name}")
        initial_rows = len(df)

        # Make a copy to avoid modifying original
        df = df.copy()

        # 1. Standardize column names
        df = self._standardize_columns(df)

        # 2. Handle missing values
        df = self._handle_missing_values(df)

        # 3. Remove duplicates
        df = self._remove_duplicates(df)

        # 4. Clean item IDs
        df = self._clean_item_ids(df)

        # 5. Clean user IDs
        df = self._clean_user_ids(df)

        # 6. Clean prices
        df = self._clean_prices(df)

        # 7. Clean units sold
        df = self._clean_units_sold(df)

        # 8. Handle outliers
        df = self._handle_outliers(df)

        # 9. Validate dates
        df = self._validate_dates(df)

        # 10. Remove invalid transactions
        df = self._remove_invalid_transactions(df)

        final_rows = len(df)
        removed_rows = initial_rows - final_rows

        self.cleaning_stats[dataset_name] = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "removed_rows": removed_rows,
            "removal_percentage": (removed_rows / initial_rows * 100) if initial_rows > 0 else 0
        }

        logger.info(
            f"Cleaning complete for {dataset_name}: "
            f"{initial_rows} -> {final_rows} rows "
            f"({removed_rows} removed, {self.cleaning_stats[dataset_name]['removal_percentage']:.2f}%)"
        )

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase with underscores"""
        # Map common column name variations
        column_mapping = {
            "Pos Number": "pos_number",
            "Ticket Number": "ticket_number",
            "Ticket Datetime": "ticket_datetime",
            "Ticket Amount": "ticket_amount",
            "User ID": "user_id",
            "Item Id": "item_id",
            "Units Sold": "units_sold",
            "Item Price": "item_price",
            "loyalty_id": "user_id",
            "plu": "item_id",
        }

        df = df.rename(columns=column_mapping)

        # Also lowercase any remaining columns
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Log missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.debug(f"Missing values found: {missing[missing > 0].to_dict()}")

        # Drop rows with missing critical fields
        critical_columns = ["user_id", "item_id", "ticket_number"]
        for col in critical_columns:
            if col in df.columns:
                before = len(df)
                df = df.dropna(subset=[col])
                after = len(df)
                if before != after:
                    logger.debug(f"Dropped {before - after} rows due to missing {col}")

        # Fill numeric columns with appropriate defaults
        if "units_sold" in df.columns:
            df["units_sold"] = df["units_sold"].fillna(1)

        if "item_price" in df.columns:
            df["item_price"] = df["item_price"].fillna(df["item_price"].median())

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicate rows"""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} duplicate rows")

        return df

    def _clean_item_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate item IDs"""
        if "item_id" not in df.columns:
            return df

        # Convert to string
        df["item_id"] = df["item_id"].astype(str)

        # Strip whitespace
        df["item_id"] = df["item_id"].str.strip()

        # Remove empty or invalid IDs
        before = len(df)
        df = df[df["item_id"].notna() & (df["item_id"] != "") & (df["item_id"] != "nan")]
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} rows with invalid item_id")

        return df

    def _clean_user_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate user IDs"""
        if "user_id" not in df.columns:
            return df

        # Convert to string
        df["user_id"] = df["user_id"].astype(str)

        # Strip whitespace
        df["user_id"] = df["user_id"].str.strip()

        # Remove empty or invalid IDs
        before = len(df)
        df = df[df["user_id"].notna() & (df["user_id"] != "") & (df["user_id"] != "nan")]
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} rows with invalid user_id")

        return df

    def _clean_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price values"""
        if "item_price" not in df.columns:
            return df

        # Convert to numeric
        df["item_price"] = pd.to_numeric(df["item_price"], errors="coerce")

        # Remove negative prices
        before = len(df)
        df = df[df["item_price"] >= 0]
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} rows with negative prices")

        return df

    def _clean_units_sold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean units sold values"""
        if "units_sold" not in df.columns:
            return df

        # Convert to numeric
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")

        # Handle negative units (returns) - keep them but flag
        # For recommendation purposes, we focus on positive purchases
        before = len(df)
        df = df[df["units_sold"] > 0]
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} rows with non-positive units_sold")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns"""
        # Price outliers - cap at 99th percentile
        if "item_price" in df.columns:
            upper_limit = df["item_price"].quantile(0.99)
            df["item_price"] = df["item_price"].clip(upper=upper_limit)

        # Units sold outliers - cap at 99th percentile
        if "units_sold" in df.columns:
            upper_limit = df["units_sold"].quantile(0.99)
            df["units_sold"] = df["units_sold"].clip(upper=upper_limit)

        return df

    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate datetime values"""
        if "ticket_datetime" not in df.columns:
            return df

        # Convert to datetime
        df["ticket_datetime"] = pd.to_datetime(df["ticket_datetime"], errors="coerce")

        # Remove rows with invalid dates
        before = len(df)
        df = df.dropna(subset=["ticket_datetime"])
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} rows with invalid dates")

        # Remove future dates (allow some buffer for timezone differences)
        current_time = pd.Timestamp.now() + pd.Timedelta(days=1)
        before = len(df)
        df = df[df["ticket_datetime"] <= current_time]
        after = len(df)

        if before != after:
            logger.debug(f"Removed {before - after} rows with future dates")

        return df

    def _remove_invalid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove transactions that don't make business sense"""
        # Remove transactions with 0 price and 0 units
        if "item_price" in df.columns and "units_sold" in df.columns:
            before = len(df)
            df = df[~((df["item_price"] == 0) & (df["units_sold"] == 0))]
            after = len(df)

            if before != after:
                logger.debug(f"Removed {before - after} zero-value transactions")

        return df

    def get_cleaning_report(self) -> dict:
        """Get a report of all cleaning operations performed"""
        return self.cleaning_stats

    @staticmethod
    def create_user_item_matrix(
        df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        value_col: Optional[str] = "units_sold",
        binary: bool = False
    ) -> pd.DataFrame:
        """
        Create a user-item interaction matrix.

        Args:
            df: Transaction DataFrame
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            value_col: Column for interaction values (None for count)
            binary: If True, convert to binary (1/0)

        Returns:
            User-item matrix as DataFrame
        """
        if value_col and value_col in df.columns:
            matrix = df.pivot_table(
                index=user_col,
                columns=item_col,
                values=value_col,
                aggfunc="sum",
                fill_value=0
            )
        else:
            # Count interactions
            matrix = df.groupby([user_col, item_col]).size().unstack(fill_value=0)

        if binary:
            matrix = (matrix > 0).astype(int)

        return matrix
