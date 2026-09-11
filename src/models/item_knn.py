import pandas as pd
from src.config import TOP_K, KNN_NEIGHBORS
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_user_item_matrix(train_df: pd.DataFrame):
    """
    Build a sparse binary user-item interaction matrix from training data.

    Each row represents a user, each column represents an item, and a value
    of 1 indicates that the user interacted with the item during training.
    """

    # Keep only user-item pairs and remove repeated interactions
    interactions = train_df[
        ["User_ID", "Item_ID"]
    ].drop_duplicates()

    # Get unique user and item IDs
    user_ids = interactions["User_ID"].unique()
    item_ids = interactions["Item_ID"].unique()

    # Map original user IDs to numeric row indices
    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    # Map original item IDs to numeric column indices
    item_to_index = {
        item_id: index
        for index, item_id in enumerate(item_ids)
    }

    # Create the reverse mapping from matrix column index to original item ID
    index_to_item = {
        index: item_id
        for item_id, index in item_to_index.items()
    }

    # Convert user and item IDs into matrix coordinates
    rows = interactions["User_ID"].map(user_to_index)
    cols = interactions["Item_ID"].map(item_to_index)

    # Every remaining user-item pair represents one interaction
    values = [1] * len(interactions)

    # Build the sparse user-item matrix
    user_item_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(item_ids))
    )

    return user_item_matrix, user_to_index, index_to_item

def fit_item_knn(user_item_matrix):
    """
    Fit an item-item nearest-neighbor model using cosine distance.

    The user-item matrix is transposed so that each row represents
    an item through its interactions with users.
    """

    # Transpose the matrix so that each row represents an item
    item_user_matrix = user_item_matrix.T

    # Create a nearest-neighbor model based on cosine distance
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    # Fit the model on item interaction vectors
    knn_model.fit(item_user_matrix)

    return knn_model, item_user_matrix

def recommend_item_knn(
    user_id: str,
    user_item_matrix,
    user_to_index: dict,
    index_to_item: dict,
    knn_model,
    item_user_matrix,
    k: int = TOP_K
) -> list[str]:
    """
    Generate top-k item-item kNN recommendations for a user.

    Candidate items are scored by accumulating their cosine similarities
    to the items previously seen by the user. Previously seen items are
    excluded from the final recommendations.
    """

    # Find the user's row in the user-item matrix
    user_index = user_to_index[user_id]

    # Get the indices of all items previously seen by the user
    seen_item_indices = user_item_matrix[user_index].indices

    # Store seen item indices for efficient membership checks
    seen_item_set = set(seen_item_indices)

    # Find nearest neighbors for all items previously seen by the user
    distances, neighbor_indices = knn_model.kneighbors(
        item_user_matrix[seen_item_indices],
        n_neighbors=KNN_NEIGHBORS + 1
    )

    # Store the accumulated similarity score for each candidate item
    candidate_scores = {}

    # Process the neighbors of every item seen by the user
    for item_neighbors, item_distances in zip(
        neighbor_indices,
        distances
    ):
        # Skip the first neighbor because it is the item itself
        for neighbor_index, distance in zip(
            item_neighbors[1:],
            item_distances[1:]
        ):
            # Only consider items the user has not previously seen
            if neighbor_index not in seen_item_set:
                similarity = 1 - distance

                # Add the similarity to the candidate's accumulated score
                candidate_scores[neighbor_index] = (
                    candidate_scores.get(neighbor_index, 0)
                    + similarity
                )

    # Rank candidate items from highest to lowest accumulated score
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep only the top-k highest-scoring candidates
    top_candidates = sorted_candidates[:k]

    # Convert matrix item indices back to original item IDs
    recommendations = [
        index_to_item[item_index]
        for item_index, _ in top_candidates
    ]

    return recommendations

def generate_item_knn_recommendations_for_users(
    user_ids,
    user_item_matrix,
    user_to_index,
    index_to_item,
    knn_model,
    item_user_matrix,
    k: int = TOP_K
) -> dict[str, list[str]]:
    """
    Generate item-item kNN recommendations for multiple users.
    """

    recommendations_by_user = {}

    # Generate recommendations separately for each user
    for user_id in user_ids:
        recommendations_by_user[user_id] = recommend_item_knn(
            user_id,
            user_item_matrix,
            user_to_index,
            index_to_item,
            knn_model,
            item_user_matrix,
            k=k
        )

    return recommendations_by_user