
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import cosine_similarity


class SVDRecommender:
    """
    SVD-based Recommender using Surprise library
    Provides:
    - Rating prediction
    - Top-K recommendation
    - Item embeddings
    - Item similarity
    """

    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all
        )
        self.trainset = None

    def fit(self, train_df: pd.DataFrame):
        """
        Train SVD model on full dataset
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["user_id", "item_id", "rating"]],
            reader
        )

        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        """
        return self.model.predict(user_id, item_id).est

    def recommend(self, user_id, k=10):
        """
        Recommend top-k items for a given user
        """
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            # Cold start user
            return []

        seen_items = {
            self.trainset.to_raw_iid(iid)
            for (iid, _) in self.trainset.ur[inner_uid]
        }

        predictions = []

        for inner_iid in self.trainset.all_items():
            raw_iid = self.trainset.to_raw_iid(inner_iid)

            if raw_iid not in seen_items:
                est = self.model.predict(user_id, raw_iid).est
                predictions.append((raw_iid, est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in predictions[:k]]

    # ===============================
    #  NEW: Item Embedding Methods
    # ===============================

    def get_item_vector(self, item_id):
        """
        Return latent vector representation of an item
        """
        try:
            inner_iid = self.trainset.to_inner_iid(item_id)
            return self.model.qi[inner_iid]
        except ValueError:
            return None

    def item_similarity(self, item_id_1, item_id_2):
        """
        Compute cosine similarity between two items
        """
        vec1 = self.get_item_vector(item_id_1)
        vec2 = self.get_item_vector(item_id_2)

        if vec1 is None or vec2 is None:
            return 0.0

        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)

        return cosine_similarity(vec1, vec2)[0][0]