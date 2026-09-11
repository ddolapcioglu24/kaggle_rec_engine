import joblib
import pandas as pd

from src.config import PROJECT_ROOT, TOP_K
from src.data import load_data
from src.split import temporal_split
from src.baselines import recommend_most_popular
from src.models.lightgbm_model import recommend_lightgbm


# Define the path of the saved final LightGBM model
MODEL_PATH = (
    PROJECT_ROOT
    / "cache"
    / "models"
    / "lgbm_ratio10.joblib"
)

def load_training_data() -> pd.DataFrame:
    """
    Load the dataset and reconstruct the same temporal training split
    used during model development.
    """

    # Load and preprocess the raw dataset
    df = load_data()

    # Recreate the original temporal train/test split
    train_df, _ = temporal_split(df)

    return train_df

def load_model():
    """
    Load the persisted final LightGBM model from disk.
    """

    # Load the fitted LightGBM model
    model = joblib.load(MODEL_PATH)

    return model

def recommend(
    user_id: str,
    k: int = TOP_K
) -> pd.DataFrame:
    """
    Return the top-k recommended items for one user, ranked.
    """

    # Load the historical training data
    train_df = load_training_data()

    # Load the fitted LightGBM model from disk
    model = load_model()

    # Check whether the user exists in the training data
    known_users = set(train_df["User_ID"].unique())

    if user_id not in known_users:
        # Use the Most Popular recommender as fallback for unknown users
        fallback_items = recommend_most_popular(
            train_df=train_df,
            user_id=user_id,
            k=k
        )

        # Build a tidy ranked dataframe
        return pd.DataFrame({
            "user_id": [user_id] * len(fallback_items),
            "item_id": fallback_items,
            "score": [None] * len(fallback_items),
            "rank": range(1, len(fallback_items) + 1)
        })

    # Recover the feature names used by the fitted model
    feature_columns = model.feature_name_

    # Generate LightGBM recommendations for a known user
    recommendations = recommend_lightgbm(
        user_id=user_id,
        train_df=train_df,
        model=model,
        feature_columns=feature_columns,
        k=k
    )

    # Return the required serving format
    return pd.DataFrame({
        "user_id": [user_id] * len(recommendations),
        "item_id": recommendations["Item_ID"].values,
        "score": recommendations["score"].values,
        "rank": recommendations["rank"].values
    })
