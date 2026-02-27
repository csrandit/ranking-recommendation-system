import numpy as np


def precision_at_k(recommended_items, relevant_items, k):
    """
    Compute Precision@K
    """
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    hits = sum(1 for item in recommended_at_k if item in relevant_set)

    return hits / k


def recall_at_k(recommended_items, relevant_items, k):
    """
    Compute Recall@K
    """
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    hits = sum(1 for item in recommended_at_k if item in relevant_set)

    if len(relevant_set) == 0:
        return 0.0

    return hits / len(relevant_set)


def average_precision_at_k(recommended_items, relevant_items, k):
    """
    Compute Average Precision@K
    """
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    score = 0.0
    hits = 0

    for i, item in enumerate(recommended_at_k):
        if item in relevant_set:
            hits += 1
            score += hits / (i + 1)

    if hits == 0:
        return 0.0

    return score / min(len(relevant_set), k)


def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Compute NDCG@K
    """
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    dcg = 0.0
    for i, item in enumerate(recommended_at_k):
        if item in relevant_set:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg