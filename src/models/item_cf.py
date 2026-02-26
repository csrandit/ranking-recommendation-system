
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering Recommender
    """

    def __init__(self, k_similar: int = 20):
        self.k_similar = k_similar
        self.user_item_matrix = None
        self.item_similarity = None

    def fit(self, train_df: pd.DataFrame):
        """
        Build user-item matrix and compute item-item similarity.
        """

        # Build user-item matrix
        self.user_item_matrix = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        ).fillna(0)

        # Transpose to compute item-item similarity
        item_matrix = self.user_item_matrix.T

        similarity_matrix = cosine_similarity(item_matrix)

        self.item_similarity = pd.DataFrame(
            similarity_matrix,
            index=item_matrix.index,
            columns=item_matrix.index
        )

    def recommend(self, user_id: int, k: int = 10):
        """
        Recommend top-k items for a given user.
        """

        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]

        scores = {}

        for item_id, rating in rated_items.items():

            similar_items = self.item_similarity[item_id] \
                .drop(item_id) \
                .nlargest(self.k_similar)

            for sim_item, sim_score in similar_items.items():
                if user_ratings[sim_item] == 0:  # not rated yet
                    scores.setdefault(sim_item, 0)
                    scores[sim_item] += sim_score * rating

        if not scores:
            return []

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [item for item, _ in ranked_items[:k]]