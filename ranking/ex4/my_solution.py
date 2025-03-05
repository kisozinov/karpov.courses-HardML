import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.4, ndcg_top_k: int = 10,
                 subsample: float = 0.8, colsample_bytree: float = 0.6,
                 max_depth: int = 8, min_samples_leaf: int = 85):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees: List[DecisionTreeRegressor] = []
        self.feature_indices: List[np.ndarray] = []
        self.best_ndcg = 0
        self.best_ndcg_idx = -1


    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        
        self.X_train = torch.FloatTensor(X_train)
        self.ys_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_test = torch.FloatTensor(y_test).reshape(-1, 1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for qid in set(inp_query_ids):
            mask = (inp_query_ids == qid)
            inp_feat_array[mask] = StandardScaler().fit_transform(inp_feat_array[mask])

        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        np.random.seed(cur_tree_idx)    
        query_ids_train = torch.tensor(self.query_ids_train)
        unique_ids, inverse_indices = torch.unique(query_ids_train, return_inverse=True)

        # Compute lambas per query group
        lambdas = torch.zeros(self.ys_train.shape, dtype=torch.float32)
        for i in range(len(unique_ids)):
            mask = (inverse_indices == i)
            query_lambda = self._compute_lambdas(self.ys_train[mask], train_preds[mask])
            lambdas[mask] = query_lambda

        # Feature slice
        num_features = self.X_train.shape[1]
        num_selected_features = max(1, round(self.colsample_bytree * num_features))
        feature_indices = np.random.choice(num_features, num_selected_features, replace=False)

        # Object slice
        num_samples = self.X_train.shape[0]
        num_selected_samples = max(1, round(self.subsample * num_samples))
        sample_indices = np.random.choice(num_samples, num_selected_samples, replace=False)
        
        # Sampiling
        X_subset = self.X_train[sample_indices][:, feature_indices]
        y_subset = -lambdas[sample_indices]

        # Fit tree
        tree = DecisionTreeRegressor(
            random_state=cur_tree_idx,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        tree.fit(X_subset, y_subset)

        return tree, feature_indices

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        queries_list = torch.tensor(queries_list)
        unique_ids, inverse_indices = torch.unique(queries_list, return_inverse=True)
        ndcg = 0
        for i in range(len(unique_ids)):
            mask = (inverse_indices == i)
            ndcg += self._ndcg_k(true_labels[mask].flatten(), preds[mask].flatten(), self.ndcg_top_k)
        return ndcg / len(unique_ids)

    def fit(self):
        np.random.seed(0)
        y_train_preds = torch.zeros(self.ys_train.shape)
        y_test_preds = torch.zeros(self.ys_test.shape)
        for tree_idx in tqdm(range(self.n_estimators)):
            # Train one tree
            tree, feature_inds = self._train_one_tree(tree_idx, y_train_preds)
            self.trees.append(tree)
            self.feature_indices.append(feature_inds)

            # Update train and test predictions
            train_preds = torch.FloatTensor(tree.predict(self.X_train[:, feature_inds])).reshape(-1, 1)
            y_train_preds += self.lr * train_preds
            test_preds = torch.FloatTensor(tree.predict(self.X_test[:, feature_inds])).reshape(-1, 1)
            y_test_preds += self.lr * test_preds

            # Calc NDCG
            curr_ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, y_test_preds)
            if curr_ndcg > self.best_ndcg:
                self.best_ndcg = curr_ndcg
                self.best_ndcg_idx = tree_idx

            print(f"Tree {tree_idx + 1}/{self.n_estimators}, NDCG: {curr_ndcg:.4f}")

        # Remove all trees after the best one
        self.trees = self.trees[:self.best_ndcg_idx + 1]
        self.feature_indices = self.feature_indices[:self.best_ndcg_idx + 1]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.FloatTensor(torch.zeros(data.shape[0])).reshape(-1, 1)
        for tree, feature_inds in tqdm(zip(self.trees, self.feature_indices)):
            preds += self.lr * torch.FloatTensor(tree.predict(data[:, feature_inds])).reshape(-1, 1)
        return preds
        

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # рассчитаем нормировку, IdealDCG
        ideal_dcg = self._dcg_k(y_true, y_true, len(y_true))
        if ideal_dcg == 0:
            N = 0
            # return torch.zeros_like(y_true)
        else:
            N = 1 / ideal_dcg
        
        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, dim=0)
        rank_order += 1
        
        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))
            
            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true)
            
            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update =  (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)
            
            return lambda_update

    def _compute_labels_in_batch(self, y_true):
        
        # разница релевантностей каждого с каждым объектом
        rel_diff = y_true - y_true.t()
        
        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)
        
        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij

    def _compute_gain_diff(self, y_true, gain_scheme="exp2"):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        current_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        if ideal_dcg == 0:
            return 0
        return current_dcg / ideal_dcg

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, top_k: int) -> float:    
        _, indices = torch.sort(ys_pred, descending=True)
        ind = min((len(ys_true), top_k))
        ys_true_sorted = ys_true[indices][:ind]
        gain = 0
        for i, y in enumerate(ys_true_sorted, start=1):
            gain += (2 ** y.item() - 1) / math.log2(i + 1)

        return gain

    def save_model(self, path: str):
        state = {
            "trees": self.trees,
            "tree_feat": self.feature_indices,
            "lr": self.lr,
            "prune_ind": self.best_ndcg_idx
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.trees = state["trees"]
        self.feature_indices = state["tree_feat"]
        self.lr = state["lr"]
        self.best_ndcg_idx = state["prune_ind"]
