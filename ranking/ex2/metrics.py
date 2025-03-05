from math import log2

import torch
from torch import Tensor, sort, where, arange


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    sorted_indices = ys_pred.argsort(descending=True)
    sorted_true = ys_true[sorted_indices]
    
    swapped_pairs = 0
    n = len(sorted_true)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sorted_true[i] < sorted_true[j]:
                swapped_pairs += 1
                
    return swapped_pairs


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == "exp2":
        return 2 ** y_value - 1
    else:
        return y_value

def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:    
    _, order = sort(ys_pred, descending=True)
    # res = 0
    index = arange(len(order), dtype=torch.float64) + 1
    return (compute_gain(ys_true[order], gain_scheme) / torch.log2(index + 1)).sum().item()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    current_dcg = dcg(ys_true, ys_pred, gain_scheme)
    ideal_dcg = dcg(ys_true, ys_true, gain_scheme)
    return current_dcg / ideal_dcg


def precision_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if 1 not in ys_true:
        return -1
    
    _, k_pred_indices = sort(ys_pred, descending=True)
    k_pred_indices = k_pred_indices[:k]
    # print("k_pred_indices=", k_pred_indices)
    num_retrieved = k_pred_indices.shape[0]
    # print("k=", k)
    num_relevant = (ys_true[k_pred_indices] == 1).sum().item()
    total_rel = (ys_true == 1).sum().item()

    if num_retrieved > total_rel:
        return num_relevant / total_rel
    return num_relevant / num_retrieved

def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, order = sort(ys_pred, descending=True)
    true_sorted_by_preds = ys_true[order]
    rank = 1 + (true_sorted_by_preds == 1).nonzero(as_tuple=True)[0].item()
    return 1 / rank

def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    _, pred_indices = sort(ys_pred, descending=True)
    p_rels = ys_true[pred_indices]
    p_look = 1
    pfound = p_look * p_rels[0].item()
    for i in range(1, len(pred_indices)):
        # p_rel = p_rels[i].item()
        p_look = p_look * (1 - p_rels[i-1].item()) * (1 - p_break)
        pfound += p_look * p_rels[i].item()
    return pfound


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    _, pred_indices = sort(ys_pred, descending=True)
    num_relevant = 0
    num_retrieved = 0
    moving_sum = 0

    for idx in pred_indices:
        num_retrieved += 1

        if ys_true[idx] == 0:
            continue
        
        num_relevant += ys_true[idx]
        moving_sum += num_relevant / num_retrieved
    
    # print((ys_true == 1).sum().item())
    return moving_sum / (ys_true == 1).sum().item()
