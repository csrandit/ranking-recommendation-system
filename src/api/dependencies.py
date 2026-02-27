from functools import lru_cache
from src.config import settings
from src.models.user_cf import UserBasedCF
from src.data_preprocessing import MovieLensDataLoader


@lru_cache()
def get_model_and_movies():
    loader = MovieLensDataLoader(settings.DATA_PATH)
    ratings = loader.load_ratings()
    train_df, _ = loader.train_test_split(ratings)

    model = UserBasedCF(k_neighbors=40)
    model.fit(train_df)

    movies_df = loader.load_movies()
    movie_dict = dict(zip(movies_df.item_id, movies_df.title))

    return model, movie_dict