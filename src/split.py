import pandas as pd

from src.config import SPLIT_DATE


def temporal_split(df: pd.DataFrame):
    """
    Split interactions chronologically into training and test sets.

    Interactions before the configured split date are used for training,
    while interactions on or after the split date are reserved for testing.
    """

    # Convert the configured split date to a pandas timestamp
    split_date = pd.Timestamp(SPLIT_DATE)

    # Split interactions by time to avoid future-data leakage
    train_df = df[df["Timestamp"] < split_date].copy()
    test_df = df[df["Timestamp"] >= split_date].copy()

    return train_df, test_df


def build_evaluation_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build an evaluation set containing only previously unseen user-item pairs.

    Test interactions whose user-item pair already appeared during training
    are removed because the project evaluates recommendations of unseen items.
    """

    # Store all user-item pairs observed during training
    train_pairs = set(
        zip(train_df["User_ID"], train_df["Item_ID"])
    )

    # Identify test interactions that were already observed in training
    test_seen_before = test_df.apply(
        lambda row: (row["User_ID"], row["Item_ID"]) in train_pairs,
        axis=1
    )

    # Keep only new user-item interactions as evaluation targets
    evaluation_df = test_df[~test_seen_before].copy()

    return evaluation_df