"""Tests for data cleaning utilities"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.utils.data_cleaner import DataCleaner


class TestDataCleaner:
    """Tests for DataCleaner class"""

    @pytest.fixture
    def cleaner(self):
        """Create cleaner instance"""
        return DataCleaner()

    @pytest.fixture
    def dirty_data(self) -> pd.DataFrame:
        """Create sample dirty data"""
        return pd.DataFrame({
            "User ID": ["1", "2", "3", None, "4", "4", "5"],
            "Item Id": ["A", "B", "", "D", "E", "E", "F"],
            "Units Sold": [1, 2, -1, 3, 4, 4, 0],
            "Item Price": [10.0, -5.0, 20.0, 30.0, None, 40.0, 50.0],
            "Ticket Number": [1, 2, 3, 4, 5, 5, 6],
            "Ticket Datetime": [
                datetime.now(),
                datetime.now(),
                datetime.now(),
                None,
                datetime.now(),
                datetime.now(),
                datetime.now() + timedelta(days=365)  # Future date
            ],
            "Pos Number": [1, 1, 1, 1, 1, 1, 1],
            "Ticket Amount": [10, 20, 30, 40, 50, 50, 60]
        })

    def test_standardize_columns(self, cleaner, dirty_data):
        """Test column name standardization"""
        cleaned = cleaner.clean_transactions(dirty_data, "test")

        # Check all columns are lowercase with underscores
        for col in cleaned.columns:
            assert col == col.lower()
            assert " " not in col

    def test_remove_missing_user_id(self, cleaner, dirty_data):
        """Test removal of rows with missing user_id"""
        cleaned = cleaner.clean_transactions(dirty_data, "test")

        assert cleaned["user_id"].isna().sum() == 0
        assert "nan" not in cleaned["user_id"].values

    def test_remove_empty_item_id(self, cleaner, dirty_data):
        """Test removal of rows with empty item_id"""
        cleaned = cleaner.clean_transactions(dirty_data, "test")

        assert "" not in cleaned["item_id"].values
        assert cleaned["item_id"].notna().all()

    def test_remove_negative_prices(self, cleaner, dirty_data):
        """Test removal of rows with negative prices"""
        cleaned = cleaner.clean_transactions(dirty_data, "test")

        assert (cleaned["item_price"] >= 0).all()

    def test_remove_negative_units(self, cleaner, dirty_data):
        """Test removal of rows with non-positive units"""
        cleaned = cleaner.clean_transactions(dirty_data, "test")

        assert (cleaned["units_sold"] > 0).all()

    def test_remove_duplicates(self, cleaner, dirty_data):
        """Test duplicate removal"""
        cleaned = cleaner.clean_transactions(dirty_data, "test")

        # Should have fewer rows than original due to duplicates
        assert len(cleaned) < len(dirty_data)

    def test_cleaning_report(self, cleaner, dirty_data):
        """Test cleaning report generation"""
        cleaner.clean_transactions(dirty_data, "test_dataset")
        report = cleaner.get_cleaning_report()

        assert "test_dataset" in report
        assert "initial_rows" in report["test_dataset"]
        assert "final_rows" in report["test_dataset"]
        assert "removed_rows" in report["test_dataset"]

    def test_create_user_item_matrix(self, sample_transactions):
        """Test user-item matrix creation"""
        matrix = DataCleaner.create_user_item_matrix(
            sample_transactions,
            user_col="user_id",
            item_col="item_id"
        )

        # Check matrix shape
        n_users = sample_transactions["user_id"].nunique()
        n_items = sample_transactions["item_id"].nunique()

        assert matrix.shape[0] == n_users
        assert matrix.shape[1] == n_items

    def test_create_binary_matrix(self, sample_transactions):
        """Test binary matrix creation"""
        matrix = DataCleaner.create_user_item_matrix(
            sample_transactions,
            user_col="user_id",
            item_col="item_id",
            binary=True
        )

        # All values should be 0 or 1
        unique_values = matrix.values.flatten()
        assert set(unique_values).issubset({0, 1})
