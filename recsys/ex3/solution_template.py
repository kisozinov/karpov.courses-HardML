from typing import List, Any

import numpy as np


def user_hitrate(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: 1 if top-k recommendations contains at lease one relevant item
    """
    return int(len(set(y_rec[:k]).intersection(set(y_rel))) > 0)


def user_precision(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of relevant items through recommendations
    """
    return len(set(y_rec[:k]).intersection(set(y_rel))) / len(set(y_rec))


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of found relevant items through recommendations
    """
    return len(set(y_rec[:k]).intersection(set(y_rel))) / len(set(y_rel))


def user_ap(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: average precision metric for user recommendations
    """
    ap = 0
    for k_ in k:
        if y_rec[k_] in y_rel:
            ap += user_precision(y_rel, y_rec, k_)
    return ap / len(y_rel)


def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: ndcg metric for user recommendations
    """
    # dcg
    rels = np.array([1 if item in y_rel else 0 for item in y_rec[:k]])
    dcg = rels[0] + np.sum(rels[1:] / np.log2(np.arange(2, k + 1)))
    # idcg
    rels = np.array([1] * len(y_rel))
    idcg = rels[0] + np.sum(rels[1:] / np.log2(np.arange(2, len(y_rel) + 1)))
    return dcg / idcg


def user_rr(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: reciprocal rank for user recommendations
    """
    rank = 1 + next((i for i, item in enumerate(y_rec[:k]) if item in y_rel), k)
    return 1 / rank
