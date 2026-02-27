from surprise import Dataset, Reader, SVD
from surprise import Dataset, Reader, SVD
import pandas as pd



class SVDRecommender:
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all
        )
        self.trainset = None

    def fit(self, train_df: pd.DataFrame):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def predict(self, user_id, item_id):
        return self.model.predict(user_id, item_id).est

    def recommend(self, user_id, k=10):
        all_items = self.trainset.all_items()
        inner_uid = self.trainset.to_inner_uid(user_id)

        seen_items = {
            self.trainset.to_raw_iid(iid)
            for (iid, _) in self.trainset.ur[inner_uid]
        }

        predictions = []

        for inner_iid in all_items:
            raw_iid = self.trainset.to_raw_iid(inner_iid)
            if raw_iid not in seen_items:
                est = self.model.predict(user_id, raw_iid).est
                predictions.append((raw_iid, est))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in predictions[:k]]