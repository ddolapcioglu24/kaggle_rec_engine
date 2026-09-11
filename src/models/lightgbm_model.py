import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import (
    RANDOM_SEED,
    TOP_K,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_NUM_LEAVES
)

def add_lightgbm_features(
    examples_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add LightGBM features to user-item pairs using training data only.

    All aggregate statistics are computed exclusively from the training
    period to avoid future-data leakage.
    """
    # Work on a copy to avoid modifying the original dataframe
    featured_df = examples_df.copy()

    # Use the latest training timestamp as the end of the observed training period
    training_reference_date = train_df["Timestamp"].max()

    # Calculate each user's tenure from their first observed interaction
    user_first_interactions = (
        train_df.groupby("User_ID")["Timestamp"]
        .min()
    )

    user_tenure_days = (
        training_reference_date - user_first_interactions
    ).dt.days

    # Add user tenure to each user-item pair
    featured_df["user_tenure_days"] = (
        featured_df["User_ID"]
        .map(user_tenure_days)
    )

    # Compute user and item interaction counts from training data
    user_interaction_counts = (
        train_df.groupby("User_ID")
        .size()
    )

    item_interaction_counts = (
        train_df.groupby("Item_ID")
        .size()
    )

    # Add activity features to each user-item pair
    featured_df["user_interaction_count"] = (
        featured_df["User_ID"]
        .map(user_interaction_counts)
    )

    featured_df["item_interaction_count"] = (
        featured_df["Item_ID"]
        .map(item_interaction_counts)
    )

    # Compute historical average ratings from training data
    user_mean_ratings = (
        train_df.groupby("User_ID")["Rating"]
        .mean()
    )

    item_mean_ratings = (
        train_df.groupby("Item_ID")["Rating"]
        .mean()
    )

    # Add historical rating features
    featured_df["user_mean_rating"] = (
        featured_df["User_ID"]
        .map(user_mean_ratings)
    )

    featured_df["item_mean_rating"] = (
        featured_df["Item_ID"]
        .map(item_mean_ratings)
    )

    # Add a user-item rating compatibility feature
    featured_df["rating_difference"] = abs(
        featured_df["user_mean_rating"]
        - featured_df["item_mean_rating"]
    )

    # Compute historical average prices from training data
    user_mean_prices = (
        train_df.groupby("User_ID")["Price"]
        .mean()
    )

    item_mean_prices = (
        train_df.groupby("Item_ID")["Price"]
        .mean()
    )

    # Add historical price features
    featured_df["user_mean_price"] = (
        featured_df["User_ID"]
        .map(user_mean_prices)
    )

    featured_df["item_mean_price"] = (
        featured_df["Item_ID"]
        .map(item_mean_prices)
    )

    # Add a user-item price compatibility feature
    featured_df["price_difference"] = abs(
        featured_df["user_mean_price"]
        - featured_df["item_mean_price"]
    )

    return featured_df

def sample_negative_examples(
    train_df: pd.DataFrame,
    negative_ratio: int = 1
) -> pd.DataFrame:
    """
    Sample unseen user-item pairs as negative training examples.

    The number of sampled negatives is controlled by negative_ratio.
    For example, negative_ratio=3 creates three negative examples
    for each positive user-item interaction.
    """

    # Get the catalog of items observed during training
    all_items = train_df["Item_ID"].unique()

    # Create a reproducible random number generator
    rng = np.random.default_rng(RANDOM_SEED)

    # Store the items previously seen by each user
    seen_items_by_user = (
        train_df.groupby("User_ID")["Item_ID"]
        .apply(set)
        .to_dict()
    )

    # Get unique positive user-item pairs
    positive_pairs = (
        train_df[["User_ID", "Item_ID"]]
        .drop_duplicates()
    )

    # Count the number of positive interactions for each user
    positive_counts_by_user = (
        positive_pairs.groupby("User_ID")
        .size()
    )

    # Store sampled negative pairs
    negative_examples = []

    # Sample the requested number of unique negative items for each user
    for user_id, num_positives in positive_counts_by_user.items():
        seen_items = seen_items_by_user[user_id]

        # Find items that the user has not seen during training
        unseen_items = [
            item
            for item in all_items
            if item not in seen_items
        ]

        # Sample unique unseen items without replacement
        sampled_items = rng.choice(
            unseen_items,
            size=num_positives * negative_ratio,
            replace=False
        )

        # Store each sampled pair as a negative example
        for item_id in sampled_items:
            negative_examples.append(
                (user_id, item_id, 0)
            )

    # Convert the sampled negative pairs into a DataFrame
    negative_df = pd.DataFrame(
        negative_examples,
        columns=["User_ID", "Item_ID", "label"]
    )

    return negative_df

def fit_lightgbm_model(
    training_examples: pd.DataFrame,
    feature_columns: list[str]
) -> LGBMClassifier:
    """
    Train a LightGBM binary classifier on engineered user-item features.

    The model learns to distinguish observed interactions from sampled
    unseen user-item pairs.
    """
    # Separate input features from the binary target
    X = training_examples[feature_columns]
    y = training_examples["label"]

    # Create the LightGBM classifier
    model = LGBMClassifier(
        objective="binary",
        n_estimators=LIGHTGBM_N_ESTIMATORS,
        learning_rate=LIGHTGBM_LEARNING_RATE,
        num_leaves=LIGHTGBM_NUM_LEAVES,
        random_state=RANDOM_SEED
    )

    # Train the model
    model.fit(X, y)

    return model

def recommend_lightgbm(
    user_id: str,
    train_df: pd.DataFrame,
    model: LGBMClassifier,
    feature_columns: list[str],
    k: int = TOP_K
) -> pd.DataFrame:
    """
    Generate top-k LightGBM recommendations for one user.
    """
    # Get the full item catalogue observed during training
    all_items = train_df["Item_ID"].unique()

    # Collect items already seen by the user
    seen_items = set(
        train_df.loc[
            train_df["User_ID"] == user_id,
            "Item_ID"
        ]
    )

    # Keep only items the user has not previously interacted with
    candidate_items = [
        item
        for item in all_items
        if item not in seen_items
    ]

    # Create one user-item pair for each candidate item
    candidate_df = pd.DataFrame({
        "User_ID": [user_id] * len(candidate_items),
        "Item_ID": candidate_items
    })

    # Add the same features used during model training
    candidate_df = add_lightgbm_features(
        candidate_df,
        train_df
    )

    # Predict the interaction probability for each candidate item
    candidate_df["score"] = model.predict_proba(
        candidate_df[feature_columns]
    )[:, 1]

    # Rank candidate items from highest to lowest predicted probability
    recommendations = (
        candidate_df
        .sort_values(
            by="score",
            ascending=False
        )
        .head(k)
        .copy()
    )

    # Add ranking positions starting from 1
    recommendations["rank"] = range(
        1,
        len(recommendations) + 1
    )

    return recommendations

def prepare_lightgbm_feature_aggregates(
    train_df: pd.DataFrame
) -> dict:
    """
    Precompute training-derived aggregates used during batch scoring.
    """

    # Use the latest training timestamp as the reference date
    training_reference_date = train_df["Timestamp"].max()

    # Compute user tenure
    user_first_interactions = (
        train_df.groupby("User_ID")["Timestamp"]
        .min()
    )

    user_tenure_days = (
        training_reference_date - user_first_interactions
    ).dt.days

    # Compute user and item interaction counts
    user_interaction_counts = (
        train_df.groupby("User_ID")
        .size()
    )

    item_interaction_counts = (
        train_df.groupby("Item_ID")
        .size()
    )

    # Compute historical mean ratings
    user_mean_ratings = (
        train_df.groupby("User_ID")["Rating"]
        .mean()
    )

    item_mean_ratings = (
        train_df.groupby("Item_ID")["Rating"]
        .mean()
    )

    # Compute historical mean prices
    user_mean_prices = (
        train_df.groupby("User_ID")["Price"]
        .mean()
    )

    item_mean_prices = (
        train_df.groupby("Item_ID")["Price"]
        .mean()
    )

    return {
        "user_tenure_days": user_tenure_days,
        "user_interaction_counts": user_interaction_counts,
        "item_interaction_counts": item_interaction_counts,
        "user_mean_ratings": user_mean_ratings,
        "item_mean_ratings": item_mean_ratings,
        "user_mean_prices": user_mean_prices,
        "item_mean_prices": item_mean_prices
    }


def add_lightgbm_features_from_aggregates(
    examples_df: pd.DataFrame,
    aggregates: dict
) -> pd.DataFrame:
    """
    Add LightGBM features using precomputed training aggregates.
    """

    # Work on a copy to avoid modifying the original dataframe
    featured_df = examples_df.copy()

    # Add user-level features
    featured_df["user_tenure_days"] = (
        featured_df["User_ID"]
        .map(aggregates["user_tenure_days"])
    )

    featured_df["user_interaction_count"] = (
        featured_df["User_ID"]
        .map(aggregates["user_interaction_counts"])
    )

    featured_df["user_mean_rating"] = (
        featured_df["User_ID"]
        .map(aggregates["user_mean_ratings"])
    )

    featured_df["user_mean_price"] = (
        featured_df["User_ID"]
        .map(aggregates["user_mean_prices"])
    )

    # Add item-level features
    featured_df["item_interaction_count"] = (
        featured_df["Item_ID"]
        .map(aggregates["item_interaction_counts"])
    )

    featured_df["item_mean_rating"] = (
        featured_df["Item_ID"]
        .map(aggregates["item_mean_ratings"])
    )

    featured_df["item_mean_price"] = (
        featured_df["Item_ID"]
        .map(aggregates["item_mean_prices"])
    )

    # Add user-item compatibility features
    featured_df["rating_difference"] = abs(
        featured_df["user_mean_rating"]
        - featured_df["item_mean_rating"]
    )

    featured_df["price_difference"] = abs(
        featured_df["user_mean_price"]
        - featured_df["item_mean_price"]
    )

    return featured_df

def generate_lightgbm_recommendations_for_users(
    user_ids,
    train_df,
    model,
    feature_columns,
    k: int = TOP_K,
) -> dict[str, list[str]]:
    """
    Generate LightGBM recommendations for multiple users
    using precomputed training aggregates.
    """

    # Precompute shared feature aggregates once
    aggregates = prepare_lightgbm_feature_aggregates(train_df)

    # Get the full item catalogue once
    all_items = train_df["Item_ID"].unique()

    # Store previously seen items for each user once
    seen_items_by_user = (
        train_df.groupby("User_ID")["Item_ID"]
        .apply(set)
        .to_dict()
    )

    recommendations_by_user = {}

    # Generate recommendations for each user
    for user_id in user_ids:

        # Get items already seen by this user
        seen_items = seen_items_by_user.get(
            user_id,
            set()
        )

        # Keep only unseen items as candidates
        candidate_items = [
            item
            for item in all_items
            if item not in seen_items
        ]

        # Create candidate user-item pairs
        candidate_df = pd.DataFrame({
            "User_ID": [user_id] * len(candidate_items),
            "Item_ID": candidate_items
        })

        # Add features using precomputed aggregates
        candidate_df = add_lightgbm_features_from_aggregates(
            candidate_df,
            aggregates
        )

        # Predict interaction probabilities
        candidate_df["score"] = model.predict_proba(
            candidate_df[feature_columns]
        )[:, 1]

        # Keep the top-k item IDs
        top_items = (
            candidate_df
            .sort_values(
                "score",
                ascending=False
            )
            .head(k)["Item_ID"]
            .tolist()
        )

        recommendations_by_user[user_id] = top_items

    return recommendations_by_user
