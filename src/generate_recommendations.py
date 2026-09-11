import json
import time
import joblib
import pandas as pd

from src.config import (
    PROJECT_ROOT,
    TOP_K,
    SPLIT_DATE,
    RANDOM_SEED,
    KNN_NEIGHBORS,
    LIGHTGBM_N_ESTIMATORS,
    LIGHTGBM_LEARNING_RATE,
    LIGHTGBM_NUM_LEAVES,
    FINAL_NEGATIVE_SAMPLING_RATIO
)
from src.data import load_data
from src.split import temporal_split
from src.models.lightgbm_model import (
    prepare_lightgbm_feature_aggregates,
    add_lightgbm_features_from_aggregates
)


# Paths for the persisted model and final output files
MODEL_PATH = (
    PROJECT_ROOT
    / "cache"
    / "models"
    / "lgbm_ratio10.joblib"
)

RECOMMENDATIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "recommendations.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics.json"
)

EVALUATION_METRICS_PATH = (
    PROJECT_ROOT
    / "cache"
    / "metrics"
    / "evaluation_metrics.json"
)

def generate_batch_recommendations(
    user_ids,
    train_df,
    model,
    feature_columns,
    k=TOP_K
) -> pd.DataFrame:
    """
    Generate top-k LightGBM recommendations for all users
    while preserving scores and ranks for batch export.
    """

    # Precompute shared LightGBM aggregates once
    aggregates = prepare_lightgbm_feature_aggregates(train_df)

    # Get the full item catalogue once
    all_items = train_df["Item_ID"].unique()

    # Store previously seen items for each user once
    seen_items_by_user = (
        train_df.groupby("User_ID")["Item_ID"]
        .apply(set)
        .to_dict()
    )

    batch_rows = []

    # Generate recommendations for each user
    for user_id in user_ids:

        seen_items = seen_items_by_user.get(
            user_id,
            set()
        )

        # Keep only unseen candidate items
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

        # Select the top-k recommendations
        recommendations = (
            candidate_df
            .sort_values("score", ascending=False)
            .head(k)
            .copy()
        )

        recommendations["rank"] = range(
            1,
            len(recommendations) + 1
        )

        # Store output rows
        for _, row in recommendations.iterrows():
            batch_rows.append({
                "user_id": user_id,
                "item_id": row["Item_ID"],
                "score": row["score"],
                "rank": row["rank"]
            })

    return pd.DataFrame(batch_rows)

def main():
    """
    Run the batch recommendation pipeline.
    """

    start_time = time.perf_counter()

    # Load and split the dataset
    df = load_data()
    train_df, _ = temporal_split(df)

    # Load the persisted final LightGBM model
    model = joblib.load(MODEL_PATH)

    # Recover the feature names stored in the fitted model
    feature_columns = model.feature_name_

    # Get all users observed during training
    user_ids = train_df["User_ID"].unique()

    # Generate top-k recommendations for all users
    recommendations_df = generate_batch_recommendations(
        user_ids,
        train_df,
        model,
        feature_columns,
        k=TOP_K
    )

    # Use one timestamp for the entire batch run
    generated_at = pd.Timestamp.now(tz="UTC")

    recommendations_df["generated_at"] = generated_at

    # Write the required recommendation output
    recommendations_df.to_csv(
        RECOMMENDATIONS_PATH,
        index=False
    )

    elapsed_time = time.perf_counter() - start_time

    # Load the final evaluation metrics saved by the notebook
    with open(EVALUATION_METRICS_PATH, "r") as file:
        evaluation_metrics = json.load(file)

    # Store the final evaluation results from Stage 7
    metrics_output = {
        "evaluation_metrics": evaluation_metrics,

        "selected_model": "LightGBM 1:10",

        "config": {
        "top_k": TOP_K,
        "split_date": SPLIT_DATE,
        "random_seed": RANDOM_SEED,
        "knn_neighbors": KNN_NEIGHBORS,
        "negative_sampling_ratio": FINAL_NEGATIVE_SAMPLING_RATIO,
        "lightgbm_n_estimators": LIGHTGBM_N_ESTIMATORS,
        "lightgbm_learning_rate": LIGHTGBM_LEARNING_RATE,
        "lightgbm_num_leaves": LIGHTGBM_NUM_LEAVES
        },

        "batch_runtime_seconds": elapsed_time
    }

    # Write evaluation metrics and configuration to JSON
    with open(METRICS_PATH, "w") as file:
        json.dump(
            metrics_output,
            file,
            indent=4
        )

    print("Users processed:", len(user_ids))
    print("Recommendation rows:", len(recommendations_df))
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("Saved recommendations to:", RECOMMENDATIONS_PATH)
    print("Saved metrics to:", METRICS_PATH)


if __name__ == "__main__":
    main()