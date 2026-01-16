"""Pytest configuration and fixtures"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Create sample transaction data for testing"""
    np.random.seed(42)

    n_users = 20
    n_items = 100
    n_transactions = 500

    # Generate users
    loyal_users = [f"loyal_{i}" for i in range(n_users // 2)]
    new_users = [f"new_{i}" for i in range(n_users // 2)]
    all_users = loyal_users + new_users

    # Generate items
    items = [f"item_{i}" for i in range(n_items)]
    item_prices = {item: np.random.uniform(1, 50) for item in items}

    # Generate transactions
    data = []
    base_date = datetime.now() - timedelta(days=180)

    for i in range(n_transactions):
        user = np.random.choice(all_users)
        item = np.random.choice(items)
        units = np.random.randint(1, 5)
        date = base_date + timedelta(days=np.random.randint(0, 180))

        data.append({
            "ticket_number": i // 5,  # 5 items per ticket
            "pos_number": np.random.randint(1, 3),
            "ticket_datetime": date,
            "ticket_amount": np.random.uniform(10, 200),
            "user_id": user,
            "item_id": item,
            "units_sold": units,
            "item_price": item_prices[item],
            "user_type": "loyal" if user.startswith("loyal") else "new"
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_loyal_transactions(sample_transactions: pd.DataFrame) -> pd.DataFrame:
    """Get only loyal customer transactions"""
    return sample_transactions[sample_transactions["user_type"] == "loyal"].copy()


@pytest.fixture
def sample_new_transactions(sample_transactions: pd.DataFrame) -> pd.DataFrame:
    """Get only new customer transactions"""
    return sample_transactions[sample_transactions["user_type"] == "new"].copy()


@pytest.fixture
def sample_user_history(sample_transactions: pd.DataFrame) -> pd.DataFrame:
    """Get transaction history for a specific user"""
    user_id = sample_transactions["user_id"].iloc[0]
    return sample_transactions[sample_transactions["user_id"] == user_id].copy()
