import pandas as pd
from pathlib import Path
from typing import Optional


class MovieLensDataLoader:
    """
    Responsible for loading MovieLens 100K dataset.
    Falls back to online source if local files not found.
    """

    MOVIELENS_URL_BASE = "https://files.grouplens.org/datasets/movielens/ml-100k"

    def __init__(self, data_dir: Optional[str] = None):
        """
        Parameters
        ----------
        data_dir : Optional[str]
            Local path to ml-100k folder.
            If None → data will be loaded from the official online source.
        """
        self.data_dir = Path(data_dir) if data_dir else None

    # =========================
    # Load Ratings
    # =========================
    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings from local file if exists,
        otherwise load directly from MovieLens website.
        """

        # Try local first
        if self.data_dir:
            local_path = self.data_dir / "u.data"
            if local_path.exists():
                return pd.read_csv(
                    local_path,
                    sep="\t",
                    names=["user_id", "item_id", "rating", "timestamp"],
                    engine="python"
                )

        # Fallback to online
        url = f"{self.MOVIELENS_URL_BASE}/u.data"

        return pd.read_csv(
            url,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"]
        )

    # =========================
    # Load Movies Metadata
    # =========================
    def load_movies(self) -> pd.DataFrame:

        # Try local first
        if self.data_dir:
            local_path = self.data_dir / "u.item"
            if local_path.exists():
                return pd.read_csv(
                    local_path,
                    sep="|",
                    encoding="latin-1",
                    header=None,
                    usecols=[0, 1],
                    names=["item_id", "title"]
                )

        # Fallback to online
        url = f"{self.MOVIELENS_URL_BASE}/u.item"

        return pd.read_csv(
            url,
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=[0, 1],
            names=["item_id", "title"]
        )

    # =========================
    # Train / Test Split
    # =========================
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

            if len(user_data) < 2:
                continue  # avoid users with single interaction

            shuffled_user_data = user_data.sample(
                frac=1,
                random_state=seed
            )

            split_idx = int(len(shuffled_user_data) * (1 - test_size))

            train_parts.append(shuffled_user_data.iloc[:split_idx])
            test_parts.append(shuffled_user_data.iloc[split_idx:])

        train_df = pd.concat(train_parts).reset_index(drop=True)
        test_df = pd.concat(test_parts).reset_index(drop=True)

        return (
            train_df[["user_id", "item_id", "rating"]],
            test_df[["user_id", "item_id", "rating"]],
        )


if __name__ == "__main__":
    from src.models.popularity import PopularityRecommender
    from src.evaluation.evaluator import RecommenderEvaluator
    from src.models.user_cf import UserBasedCF

    loader = MovieLensDataLoader("data/raw/ml-100k")

    full_ratings = loader.load_ratings()
    train_set, test_set = loader.train_test_split(full_ratings)

    print("\nTrain size:", len(train_set))
    print("Test size:", len(test_set))

    print("\nTrain set preview:")
    print(train_set.head())

    print("\nTest set preview:")
    print(test_set.head())

    print("\nTrain columns:", train_set.columns)

    # =========================
    # Popularity Model
    # =========================
    model = PopularityRecommender()
    model.fit(train_set)

    print("\nTop 5 popular items:", model.recommend(user_id=1, k=5))

    evaluator = RecommenderEvaluator(model, train_set, test_set)
    results = evaluator.evaluate(k=10)

    print("\nPopularity Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # =========================
    # User-Based CF
    # =========================
    print("\nTraining User-Based CF model...")

    user_cf = UserBasedCF(k_neighbors=40)
    user_cf.fit(train_set)

    evaluator_cf = RecommenderEvaluator(user_cf, train_set, test_set)
    results_cf = evaluator_cf.evaluate(k=10)

    print("\nUser-Based CF Results:")
    for metric_name, metric_value in results_cf.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # =========================
    # Item-Based CF
    # =========================
    from src.models.item_cf import ItemBasedCF

    print("\nTraining Item-Based CF model...")

    item_cf = ItemBasedCF(k_similar=20)
    item_cf.fit(train_set)

    evaluator_item = RecommenderEvaluator(item_cf, train_set, test_set)
    results_item = evaluator_item.evaluate(k=10)

    print("\nItem-Based CF Results:")
    for metric_name, metric_value in results_item.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # =========================
    # SVD Model
    # =========================
    from src.models.svd_model import SVDRecommender

    print("\nTraining SVD model...")

    svd_model = SVDRecommender(
        n_factors=100,
        n_epochs=50,
        lr_all=0.007,
        reg_all=0.02
    )
    svd_model.fit(train_set)

    evaluator_svd = RecommenderEvaluator(svd_model, train_set, test_set)
    results_svd = evaluator_svd.evaluate(k=10)

    print("\nSVD Results:")
    for metric_name, metric_value in results_svd.items():
        print(f"{metric_name}: {metric_value:.4f}")


    print("\nTotal interactions:", len(full_ratings))
    print("Train interactions:", len(train_set))
    print("Test interactions:", len(test_set))

