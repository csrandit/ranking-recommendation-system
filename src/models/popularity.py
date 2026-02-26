import pandas as pd
import numpy as np


class PopularityRecommender:
    """
    Popularity-based recommendation model.
    Recommends globally popular items.
    """

    def __init__(self):
        self.popularity_df = None

    def fit(self, train_df: pd.DataFrame):
        """
        Compute popularity scores from training data.
        """

        item_stats = train_df.groupby("item_id").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        ).reset_index()

        # Popularity score
        item_stats["popularity_score"] = (
            item_stats["avg_rating"] *
            np.log1p(item_stats["rating_count"])
        )

        self.popularity_df = item_stats.sort_values(
            by="popularity_score",
            ascending=False
        )

    def recommend(self, user_id: int, k: int = 5):
        """
        Recommend top-k popular items.
        Ignores user_id (global popularity).
        """

        if self.popularity_df is None:
            raise ValueError("Model not fitted yet.")

        return self.popularity_df.head(k)["item_id"].tolist()