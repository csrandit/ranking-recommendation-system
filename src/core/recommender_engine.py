from src.data_preprocessing import MovieLensDataLoader
from src.models.user_cf import UserBasedCF


class RecommenderEngine:

    def __init__(self, k_neighbors=40):
        self.k_neighbors = k_neighbors
        self.model = None
        self.movies_df = None

    def load(self):
        """
        Loads data, splits into train/test,
        trains the User-Based CF model.
        """
        # بدون مسار 👇
        loader = MovieLensDataLoader()

        ratings = loader.load_ratings()
        train_df, _ = loader.train_test_split(ratings)

        # Initialize model
        self.model = UserBasedCF(k_neighbors=self.k_neighbors)
        self.model.fit(train_df)

        # Load movie metadata
        self.movies_df = loader.load_movies()

    def recommend_for_new_user(self, liked_movie_ids, k=9):
        """
        Recommend movies for a new user
        based on liked movie IDs.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        return self.model.recommend_for_new_user(
            liked_movie_ids,
            k=k,
            return_scores=True
        )