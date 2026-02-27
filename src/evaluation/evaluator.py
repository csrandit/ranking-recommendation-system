import numpy as np
from collections import defaultdict

from src.evaluation.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
    ndcg_at_k,
)


class RecommenderEvaluator:
    """
    Evaluates recommender models using ranking metrics.
    """

    def __init__(self, model, train_df, test_df):
        self.model = model
        self.train_df = train_df
        self.test_df = test_df

        # Build ground truth from test set
        self.ground_truth = self._build_ground_truth()

    def _build_ground_truth(self):
        """
        Create mapping:
        user_id -> list of relevant item_ids
        """
        ground_truth = defaultdict(list)

        for _, row in self.test_df.iterrows():
            user_id = row["user_id"]
            item_id = row["item_id"]
            ground_truth[user_id].append(item_id)

        return ground_truth

    def evaluate(self, k=10):
        """
        Evaluate model across all users in test set.
        Returns dictionary of averaged metrics.
        """

        precision_scores = []
        recall_scores = []
        map_scores = []
        ndcg_scores = []

        for user_id, relevant_items in self.ground_truth.items():

            recommended_items = self.model.recommend(user_id=user_id, k=k)

            precision_scores.append(
                precision_at_k(recommended_items, relevant_items, k)
            )

            recall_scores.append(
                recall_at_k(recommended_items, relevant_items, k)
            )

            map_scores.append(
                average_precision_at_k(recommended_items, relevant_items, k)
            )

            ndcg_scores.append(
                ndcg_at_k(recommended_items, relevant_items, k)
            )

        return {
            "Precision@K": np.mean(precision_scores),
            "Recall@K": np.mean(recall_scores),
            "MAP@K": np.mean(map_scores),
            "NDCG@K": np.mean(ndcg_scores),
        }