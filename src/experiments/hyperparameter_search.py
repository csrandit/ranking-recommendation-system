from src.models.user_cf import UserBasedCF
from src.evaluation.evaluator import RecommenderEvaluator
from src.data_preprocessing import MovieLensDataLoader


def run_user_cf_search():

    print("Loading data...")
    loader = MovieLensDataLoader("data/raw/ml-100k")
    full_ratings = loader.load_ratings()
    train_set, test_set = loader.train_test_split(full_ratings)

    k_values = [10, 20, 30, 40, 50]

    best_k = None
    best_ndcg = -1

    results = []

    for k in k_values:
        print(f"\nTraining User-CF with k_neighbors={k}")

        model = UserBasedCF(k_neighbors=k)
        model.fit(train_set)

        evaluator = RecommenderEvaluator(model, train_set, test_set)
        metrics = evaluator.evaluate(k=10)

        ndcg = metrics["NDCG@K"]

        results.append((k, ndcg))

        print(f"NDCG@10 = {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_k = k

    print("\n==============================")
    print(f"Best k_neighbors: {best_k}")
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    print("==============================")

    return results


if __name__ == "__main__":
    run_user_cf_search()