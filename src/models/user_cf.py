import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """
    User-Based Collaborative Filtering Recommender
    """

    def __init__(self, k_neighbors: int = 20):
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.user_similarity = None

    def fit(self, train_df: pd.DataFrame):
        """
        Build user-item matrix and compute similarity matrix.
        """

        # Create user-item matrix
        self.user_item_matrix = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        ).fillna(0)

        # Compute cosine similarity between users
        self.user_similarity = cosine_similarity(self.user_item_matrix)

        # Convert to DataFrame for easier indexing
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

    def recommend(self, user_id: int, k: int = 10):
        """
        Recommend top-k items for a given user.
        """

        if user_id not in self.user_item_matrix.index:
            return []

        # Get similarity scores for the user
        sim_scores = self.user_similarity[user_id]

        # Remove self similarity
        sim_scores = sim_scores.drop(user_id)

        # Select top K similar users
        top_users = sim_scores.nlargest(self.k_neighbors)

        # Weighted sum of ratings
        weighted_ratings = np.dot(
            top_users.values,
            self.user_item_matrix.loc[top_users.index]
        )

        scores = pd.Series(
            weighted_ratings,
            index=self.user_item_matrix.columns
        )

        # Remove already rated items
        rated_items = self.user_item_matrix.loc[user_id]
        scores = scores[rated_items == 0]

        # Return top-k items
        return scores.sort_values(ascending=False).head(k).index.tolist()