from src.data_preprocessing import MovieLensDataLoader
from src.models.user_cf import UserBasedCF
from src.models.svd_model import SVDRecommender
from src.models.hybrid_model import HybridRecommender
from src.evaluation.evaluator import RecommenderEvaluator


def load_data():
    loader = MovieLensDataLoader("data/raw/ml-100k")
    ratings_df = loader.load_ratings()
    return loader.train_test_split(ratings_df)


def search_best_user_cf():

    train_df, test_df = load_data()

    best_k_value = None
    best_ndcg_score = 0

    for k_neighbors in [10, 20, 30, 40, 50]:

        print(f"\nTraining User-CF with k_neighbors={k_neighbors}")

        model = UserBasedCF(k_neighbors=k_neighbors)
        model.fit(train_df)

        evaluator = RecommenderEvaluator(model, train_df, test_df)
        results = evaluator.evaluate(k=10)

        current_ndcg_score = results["NDCG@K"]
        print(f"NDCG@10 = {current_ndcg_score:.4f}")

        if current_ndcg_score > best_ndcg_score:
            best_ndcg_score = current_ndcg_score
            best_k_value = k_neighbors

    print("\n==============================")
    print(f"Best k_neighbors: {best_k_value}")
    print(f"Best NDCG@10: {best_ndcg_score:.4f}")
    print("==============================")

    return best_k_value


def evaluate_hybrid(optimal_k):

    print("\nEvaluating Hybrid Model...")

    train_df, test_df = load_data()

    # Train CF
    cf_model = UserBasedCF(k_neighbors=optimal_k)
    cf_model.fit(train_df)

    # Train SVD
    svd_model = SVDRecommender()
    svd_model.fit(train_df)

    # Hybrid
    hybrid_model = HybridRecommender(
        cf_model=cf_model,
        svd_model=svd_model,
        alpha=0.7
    )

    evaluator = RecommenderEvaluator(hybrid_model, train_df, test_df)
    results = evaluator.evaluate(k=10)

    print("\nHybrid Results:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":

    optimal_k = search_best_user_cf()
    evaluate_hybrid(optimal_k)