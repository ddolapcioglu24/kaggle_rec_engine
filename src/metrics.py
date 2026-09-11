import math
from src.config import TOP_K


def precision_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int = TOP_K
) -> float:
    """
    Calculate Precision@K for one user's recommendation list.
    """

    # Keep only the first k recommended items
    top_k = recommended_items[:k]

    # Count how many recommended items are relevant
    hits = sum(
        item in relevant_items
        for item in top_k
    )

    # Return the fraction of top-k recommendations that are relevant
    return hits / k

def recall_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int = TOP_K
) -> float:
    """
    Calculate Recall@K for one user's recommendation list.
    """

    # Keep only the first k recommended items
    top_k = recommended_items[:k]

    # Count how many relevant items appear in the recommendations
    hits = sum(
        item in relevant_items
        for item in top_k
    )

    # Return the fraction of relevant items that were successfully recommended
    return hits / len(relevant_items)

def hit_rate_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int = TOP_K
) -> float:
    """
    Calculate Hit Rate@K for one user's recommendation list.
    """

    # Keep only the first k recommended items
    top_k = recommended_items[:k]

    # Return 1 if at least one recommended item is relevant, otherwise 0
    return float(
        any(item in relevant_items for item in top_k)
    )

def average_precision_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int = TOP_K
) -> float:
    """
    Calculate Average Precision@K for one user's recommendation list.
    """

    # Keep only the first k recommended items
    top_k = recommended_items[:k]

    hits = 0
    precision_sum = 0.0

    # Accumulate precision whenever a relevant item is found
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_items:
            hits += 1
            precision_sum += hits / rank

    # Normalize by the maximum number of relevant items recoverable within top-k
    return precision_sum / min(len(relevant_items), k)

def map_at_k(
    recommendations_by_user: dict[str, list[str]],
    relevant_items_by_user: dict[str, set[str]],
    k: int = TOP_K
) -> float:
    """
    Calculate MAP@K across all evaluation users.
    """

    ap_scores = []

    # Calculate AP@K for each evaluation user
    for user_id, relevant_items in relevant_items_by_user.items():
        recommended_items = recommendations_by_user[user_id]

        ap = average_precision_at_k(
            recommended_items,
            relevant_items,
            k
        )

        ap_scores.append(ap)

    # Return the mean AP@K across all evaluation users
    return sum(ap_scores) / len(ap_scores)

def ndcg_at_k(
    recommended_items: list[str],
    relevant_items: set[str],
    k: int = TOP_K
) -> float:
    """
    Calculate NDCG@K for one user's recommendation list.
    """

    # Keep only the first k recommended items
    top_k = recommended_items[:k]

    dcg = 0.0

    # Calculate DCG using the actual recommendation ranking
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_items:
            dcg += 1 / math.log2(rank + 1)

    # Calculate DCG for the ideal ranking
    ideal_hits = min(len(relevant_items), k)

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    # Normalize DCG by the ideal DCG
    return dcg / idcg

def evaluate_recommendations(
    recommendations_by_user: dict[str, list[str]],
    relevant_items_by_user: dict[str, set[str]],
    k: int = TOP_K
) -> dict[str, float]:
    """
    Evaluate recommendation quality across all evaluation users.
    """

    precision_scores = []
    recall_scores = []
    hit_scores = []
    ap_scores = []
    ndcg_scores = []

    # Calculate each metric for every evaluation user
    for user_id, relevant_items in relevant_items_by_user.items():
        recommended_items = recommendations_by_user[user_id]

        precision_scores.append(
            precision_at_k(recommended_items, relevant_items, k)
        )

        recall_scores.append(
            recall_at_k(recommended_items, relevant_items, k)
        )

        hit_scores.append(
            hit_rate_at_k(recommended_items, relevant_items, k)
        )

        ap_scores.append(
            average_precision_at_k(recommended_items, relevant_items, k)
        )

        ndcg_scores.append(
            ndcg_at_k(recommended_items, relevant_items, k)
        )

    # Aggregate per-user scores into dataset-level metrics
    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "hit_rate_at_k": sum(hit_scores) / len(hit_scores),
        "map_at_k": sum(ap_scores) / len(ap_scores),
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores),
    }

def catalogue_coverage(
    recommendations_by_user: dict[str, list[str]],
    catalogue_items
) -> float:
    """
    Calculate the fraction of catalogue items that are recommended
    to at least one user.
    """

    recommended_items = set()

    for recommendations in recommendations_by_user.values():
        recommended_items.update(recommendations)

    return len(recommended_items) / len(catalogue_items)

def popularity_bias(
    recommendations_by_user: dict[str, list[str]],
    train_df
) -> dict[str, float]:
    """
    Compare the average popularity rank of recommended items
    with the average popularity rank of the full catalogue.
    """

    # Count how many training interactions each item received
    item_popularity = (
        train_df.groupby("Item_ID")
        .size()
        .sort_values(ascending=False)
    )

    # Assign popularity ranks:
    # rank 1 = most popular item
    popularity_ranks = (
        item_popularity
        .rank(
            method="average",
            ascending=False
        )
        .to_dict()
    )

    # Collect popularity ranks of all recommended items
    recommended_ranks = []

    for recommendations in recommendations_by_user.values():
        for item_id in recommendations:
            recommended_ranks.append(
                popularity_ranks[item_id]
            )

    average_recommended_rank = (
        sum(recommended_ranks)
        / len(recommended_ranks)
    )

    average_catalogue_rank = (
        sum(popularity_ranks.values())
        / len(popularity_ranks)
    )

    return {
        "average_recommended_popularity_rank":
            average_recommended_rank,
        "average_catalogue_popularity_rank":
            average_catalogue_rank
    }

