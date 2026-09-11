import pandas as pd

from src.config import DATA_PATH


def load_data() -> pd.DataFrame:
    """
    Load the raw recommendation dataset and prepare it for analysis.

    The timestamp column is converted to datetime format so that
    time-based analysis and temporal splitting can be performed.
    """

    # Load the raw interaction data
    df = pd.read_csv(DATA_PATH)

    # Convert timestamps from text to pandas datetime objects
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df