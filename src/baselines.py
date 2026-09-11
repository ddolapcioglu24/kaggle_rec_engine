import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, TOP_K


def recommend_most_popular(
    train_df: pd.DataFrame,
    user_id: str,
    k: int = TOP_K
) -> list[str]:
    """
    Recommend the globally most popular unseen items for a user.

    Popularity is measured using interaction counts in the training set.
    Items already seen by the user are excluded from the recommendation list.
    """

    # Rank items from most to least popular
    item_popularity = (
        train_df.groupby("Item_ID")
        .size()
        .sort_values(ascending=False)
    )

    # Collect items already seen by the user during training
    seen_items = set(
        train_df.loc[
            train_df["User_ID"] == user_id,
            "Item_ID"
        ]
    )

    # Remove seen items from the popularity ranking
    unseen_popular_items = item_popularity[
        ~item_popularity.index.isin(seen_items)
    ]

    # Return the top-k unseen items
    return unseen_popular_items.head(k).index.tolist()


def recommend_random(
    train_df: pd.DataFrame,
    user_id: str,
    k: int = TOP_K,
    rng=None
) -> list[str]:
    """
    Recommend randomly selected unseen items for a user.

    This baseline ignores user preferences and item popularity, providing
    a simple reference point for later recommendation models.
    """

    # Get the item catalog observed during training
    all_items = train_df["Item_ID"].unique()

    # Collect items already seen by the user
    seen_items = set(
        train_df.loc[
            train_df["User_ID"] == user_id,
            "Item_ID"
        ]
    )

    # Candidate items are those not previously seen by the user
    unseen_items = [
        item
        for item in all_items
        if item not in seen_items
    ]

    # Create a reproducible generator only if one was not supplied
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Prevent errors if fewer than k unseen items are available
    num_recommendations = min(k, len(unseen_items))

    recommendations = rng.choice(
        unseen_items,
        size=num_recommendations,
        replace=False
    )

    return recommendations.tolist()


def generate_baseline_recommendations_for_users(
    train_df: pd.DataFrame,
    user_ids,
    recommender_function,
    k: int = TOP_K
) -> dict[str, list[str]]:
    """
    Generate recommendation lists for a collection of users.

    The supplied recommender function is applied independently to each user.
    """

    recommendations = {}

    # Create one shared random generator for the Random baseline
    rng = np.random.default_rng(RANDOM_SEED)

    # Generate one top-k recommendation list per user
    for user_id in user_ids:

        if recommender_function == recommend_random:
            recommendations[user_id] = recommender_function(
                train_df,
                user_id,
                k,
                rng
            )
        else:
            recommendations[user_id] = recommender_function(
                train_df,
                user_id,
                k
            )

    return recommendations