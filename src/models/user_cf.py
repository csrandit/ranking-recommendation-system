
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """
    User-Based Collaborative Filtering Recommender.
    """

    def __init__(self, k_neighbors=40):
        self.k_neighbors = k_neighbors
        self.user_item_matrix = None
        self.user_similarity = None
        self.train_df = None

    # ==============================
    # Train Model
    # ==============================

    def fit(self, train_df: pd.DataFrame):
        """
        Train model by building user-item matrix
        and computing cosine similarity between users.
        """

        self.train_df = train_df.copy()

        # Pivot to user-item matrix
        self.user_item_matrix = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        ).fillna(0)

        # Compute similarity between users
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    # ==============================
    # Recommend for existing user
    # ==============================

    def recommend(self, user_id, k=10):
        """
        Recommend top-k items for an existing user.
        """

        if user_id not in self.user_item_matrix.index:
            return []

        user_index = self.user_item_matrix.index.get_loc(user_id)

        similarity_scores = list(enumerate(self.user_similarity[user_index]))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Select top similar users (excluding self)
        neighbors = similarity_scores[1:self.k_neighbors + 1]

        user_ratings = self.user_item_matrix.iloc[user_index]

        scores = {}

        for neighbor_index, similarity in neighbors:
            neighbor_ratings = self.user_item_matrix.iloc[neighbor_index]

            for item_id, rating in neighbor_ratings.items():

                if user_ratings[item_id] == 0 and rating > 0:
                    if item_id not in scores:
                        scores[item_id] = 0
                    scores[item_id] += similarity * rating

        ranked_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [item for item, _ in ranked_items[:k]]

    # ==============================
    # Recommend for NEW user
    # ==============================

    def recommend_for_new_user(self, liked_item_ids, k=10, return_scores=False):
        """
        Recommend for a new user given a list of liked item_ids.
        """

        if self.user_item_matrix is None:
            raise ValueError("Model is not fitted yet.")

        # Create temporary user vector
        temp_user = pd.Series(
            0,
            index=self.user_item_matrix.columns
        )

        for item_id in liked_item_ids:
            if item_id in temp_user.index:
                temp_user[item_id] = 5  # assume strong like

        # Compute similarity with all existing users
        similarities = cosine_similarity(
            [temp_user.values],
            self.user_item_matrix.values
        )[0]

        similarity_scores = list(enumerate(similarities))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        neighbors = similarity_scores[:self.k_neighbors]

        scores = {}

        for neighbor_index, similarity in neighbors:
            neighbor_ratings = self.user_item_matrix.iloc[neighbor_index]

            for item_id, rating in neighbor_ratings.items():

                if item_id not in liked_item_ids and rating > 0:
                    if item_id not in scores:
                        scores[item_id] = 0
                    scores[item_id] += similarity * rating

        predictions = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if return_scores:
            return predictions[:k]
        else:
            return [item for item, _ in predictions[:k]]