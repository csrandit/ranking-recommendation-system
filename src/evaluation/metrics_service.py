from src.evaluation.evaluator import RecommenderEvaluator
from src.data_preprocessing import MovieLensDataLoader
from src.models.user_cf import UserBasedCF


def compute_metrics(data_path="data/raw/ml-100k"):

    loader = MovieLensDataLoader(data_path)

    ratings = loader.load_ratings()
    train_df, test_df = loader.train_test_split(ratings)

    # Build fresh model for evaluation
    model = UserBasedCF(k_neighbors=40)
    model.fit(train_df)

    evaluator = RecommenderEvaluator(model, train_df, test_df)

    results = evaluator.evaluate(k=10)

    return results