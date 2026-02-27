import pandas as pd
import numpy as np


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using cosine similarity.
    """

    def __init__(self, k_similar: int = 20):
        self.k_similar = k_similar
        self.item_similarity = None
        self.user_item_matrix = None

    def fit(self, train_df: pd.DataFrame):
        """
        Train model by building item-item similarity matrix.
        """

        # Create user-item matrix
        self.user_item_matrix = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        ).fillna(0)

        # Compute cosine similarity between items
        item_matrix = self.user_item_matrix.T.values

        norm = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10

        normalized = item_matrix / norm
        similarity = normalized @ normalized.T

        self.item_similarity = pd.DataFrame(
            similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

    def recommend(self, user_id: int, k: int = 10):
        """
        Recommend items for existing user.
        """

        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        liked_items = user_ratings[user_ratings > 0].index.tolist()

        scores = {}

        for item in liked_items:
            similar_items = self.item_similarity[item] \
                .sort_values(ascending=False)[1:self.k_similar + 1]

            for sim_item, similarity_score in similar_items.items():
                if user_ratings[sim_item] == 0:
                    scores[sim_item] = scores.get(sim_item, 0) + similarity_score

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [item for item, _ in ranked_items[:k]]