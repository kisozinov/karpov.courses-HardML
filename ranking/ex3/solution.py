import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List
from tqdm import tqdm


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        self.ys_train = torch.FloatTensor(y_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        _, last_indices = np.unique(inp_query_ids, return_index=True)
        last_indices = np.r_[last_indices, inp_query_ids.shape[0]]

        query_groups = np.split(inp_feat_array, last_indices[1:-1])
        scaler = StandardScaler()
        normalized_data = [scaler.fit_transform(group) for group in query_groups]

        return np.vstack(normalized_data)

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        ndcgs = []
        for epoch in tqdm(range(self.n_epochs)):
            self._train_one_epoch()
            ndcg = self._eval_test_set()
            print("Epoch: {}, NDCG@{}: {:.4f}".format(epoch, self.ndcg_top_k, ndcg))
            ndcgs.append(ndcg)
        return ndcgs


    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))

    def _train_one_epoch(self) -> None:
        self.model.train()
        # cur_batch = 0
        for query_id in np.unique(self.query_ids_train):
            docs_mask = (self.query_ids_train == query_id)
            batch_X = self.X_train[docs_mask]
            batch_ys = self.ys_train[docs_mask].flatten()
            
            self.optimizer.zero_grad()
            # if len(batch_X) > 0:
            ys_pred = self.model(batch_X).flatten()
            loss = self._calc_loss(batch_ys, ys_pred)
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for query_id in np.unique(self.query_ids_test):
                docs_mask = (self.query_ids_test == query_id)
                batch_X = self.X_test[docs_mask]
                batch_ys = self.ys_test[docs_mask].flatten()
                
                self.optimizer.zero_grad()
                if len(batch_X) > 0:
                    ys_pred = self.model(batch_X).flatten()
                    ndcg = self._ndcg_k(batch_ys, ys_pred, self.ndcg_top_k)
                    ndcgs.append(ndcg)
                else:
                    ndcgs.append(0)
            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        current_dcg = self._dcg(ys_true, ys_pred, "exp2", ndcg_top_k)
        ideal_dcg = self._dcg(ys_true, ys_true, "exp2", ndcg_top_k)
        if ideal_dcg == 0:
            return 0
        return current_dcg / ideal_dcg

    def _compute_gain(self, y_value: float, gain_scheme: str) -> float:
        if gain_scheme == "exp2":
            return 2 ** y_value - 1
        else:
            return y_value

    def _dcg(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str, ndcg_top_k: int) -> float:    
        _, order = torch.sort(ys_pred, descending=True)
        order = order[:ndcg_top_k]
        index = torch.arange(len(order), dtype=torch.float64) + 1
        return (self._compute_gain(ys_true[order], gain_scheme) / torch.log2(index + 1)).sum().item()
