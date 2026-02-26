import pandas as pd
from pathlib import Path


class MovieLensDataLoader:
    """
    Responsible for loading and splitting the MovieLens 100K dataset.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings data from u.data file.
        """
        file_path = self.data_dir / "u.data"

        ratings_df = pd.read_csv(
            file_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python"
        )

        return ratings_df

    @staticmethod
    def train_test_split(
        ratings_df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 42
    ):
        """
        Perform per-user train/test split.
        Ensures each user appears in both train and test.
        """

        train_parts = []
        test_parts = []

        for _, user_data in ratings_df.groupby("user_id"):
            shuffled_user_data = user_data.sample(frac=1, random_state=seed)

            split_idx = int(len(shuffled_user_data) * (1 - test_size))

            train_parts.append(shuffled_user_data.iloc[:split_idx])
            test_parts.append(shuffled_user_data.iloc[split_idx:])

        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df = pd.concat(test_parts).reset_index(drop=True)

        return train_df, test_df


if __name__ == "__main__":
    loader = MovieLensDataLoader("data/raw/ml-100k")

    full_ratings = loader.load_ratings()
    train_set, test_set = loader.train_test_split(full_ratings)

    from src.models.popularity import PopularityRecommender

    model = PopularityRecommender()
    model.fit(train_set)

    print("Top 5 popular items:", model.recommend(user_id=1, k=5))
    print("Total interactions:", len(full_ratings))
    print("Train interactions:", len(train_set))
    print("Test interactions:", len(test_set))