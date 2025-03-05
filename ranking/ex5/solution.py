import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F


# Замените пути до директорий и файлов! Можете использовать для локальной отладки.
# При проверке на сервере пути будут изменены
glue_qqp_dir = '/data/QQP/'
glove_path = '/data/glove.6B.50d.txt'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x-self.mu)**2 / (2 * self.sigma**2))


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        
        start = -1 + 1 / (self.kernel_num - 1)
        step = 2 / (self.kernel_num - 1)
        for mu in np.arange(start, 1, step):
            kernels.append(GaussianKernel(mu, self.sigma))
        kernels.append(GaussianKernel(sigma=self.exact_sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        if not self.out_layers:
            return torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))
        
        layers = []
        in_dim = self.kernel_num
        
        for out_dim in self.out_layers:
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.ReLU())
            in_dim = out_dim

        layers.append(torch.nn.Linear(in_dim, 1))  # Последний слой
        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        query = F.normalize(self.embeddings(query), p=2, dim=-1).unsqueeze(2)
        doc = F.normalize(self.embeddings(doc), p=2, dim=-1).unsqueeze(1)

        return torch.sum(query * doc, dim=-1)  # Эквивалентно cosine_similarity

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab.get(token, self.oov_val) for token in tokenized_text[:self.max_len]]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        tokens = self.preproc_func(self.idx_to_text_mapping.get(idx, ""))
        return self._tokenized_text_to_index(tokens)

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        id_q, id_l, id_r, target = self.index_pairs_or_triplets[idx]
        tokens_query = self._convert_text_idx_to_token_idxs(id_q)
        tokens_l = self._convert_text_idx_to_token_idxs(id_l)
        tokens_r = self._convert_text_idx_to_token_idxs(id_r)
        sample1 = {'query': tokens_query, 'document': tokens_l}
        sample2 = {'query': tokens_query, 'document': tokens_r}

        return sample1, sample2, target


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        id_l, id_r, target = self.index_pairs_or_triplets[idx]
        tokens_l = self._convert_text_idx_to_token_idxs(id_l)
        tokens_r = self._convert_text_idx_to_token_idxs(id_r)
        sample = {'query': tokens_l, 'document': tokens_r}

        return sample, target


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels


class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens(
            [self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()
        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(
            self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(
            self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
              self.idx_to_text_mapping_dev,
              vocab=self.vocab, oov_val=self.vocab['OOV'],
              preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'],
            'id_right': glue_df['qid2'],
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype(int)
        })
        return glue_df_fin

    def hadle_punctuation(self, inp_str: str) -> str:
        for p in string.punctuation:
            inp_str = inp_str.replace(p, ' ')
        return inp_str

    def simple_preproc(self, inp_str: str) -> List[str]:
        processed_str = self.hadle_punctuation(inp_str)
        return nltk.word_tokenize(processed_str.lower())

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        filtered_tokens = [token for token, count in vocab.items() if count >= min_occurancies]
        return filtered_tokens

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        texts = []
        for df in list_of_df:
            texts.extend([text for text in df["text_left"].values] + \
                                    [text for text in df["text_right"].values])
        tokens = []
        for text in set(texts):
            tokens += self.simple_preproc(text)
        token_counter = Counter(tokens)

        filtered_tokens = self._filter_rare_words(token_counter, min_occurancies)
        # print("All tokens: ", len(filtered_tokens), token_counter.most_common(5))
        return filtered_tokens

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        embeddings = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = values[1:]  # Оставляем в виде строк
                embeddings[word] = vector

        return embeddings

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        np.random.seed(random_seed)
        pad_token, oov_token = "PAD", "OOV"
        embeddings_vocab = self._read_glove_embeddings(file_path)
        emb_dim = len(next(iter(embeddings_vocab.values())))
        vocab = {pad_token: 0, oov_token: 1}
        uniq_inner_keys = set(inner_keys)
        oov_emb = np.random.uniform(-rand_uni_bound, rand_uni_bound, emb_dim)
        # vocab.update({token: idx for idx, token in enumerate(uniq_inner_keys, start=2)})
        # matrix = np.vstack((np.array([np.zeros(emb_dim), oov_emb]), np.zeros((len(uniq_inner_keys), emb_dim))))
        # print("matrix shape: ", matrix.shape)
        matrix = [np.zeros(emb_dim), oov_emb]
        unk_words = ['PAD', 'OOV']
        
        for idx, token in enumerate(uniq_inner_keys, start=2):
            
            if token in embeddings_vocab:
                matrix.append(np.array(embeddings_vocab[token], dtype=np.float32))
            else:
                matrix.append(oov_emb)
                unk_words.append(token)
            vocab[token] = idx
        assert len(matrix) == len(vocab)
        matrix = np.array(matrix)
        print(matrix)
            # if token in embeddings_vocab:
            #     matrix[idx] = np.array(embeddings_vocab[token], dtype=np.float32)
            # else:
            #     matrix[idx] = oov_emb
            #     unk_words.append(token)

        # unk_words = [token for token in uniq_inner_keys if token not in embeddings_vocab] + ['PAD', 'OOV']
        return matrix, vocab, unk_words

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        # print("unk: ", len(unk_words))
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def _gen_pairs(self, list1, list2):
        return ((x, y) for x in list1 for y in list2)

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int) -> List[List[Union[str, float]]]:
        # TODO: Изменить семплирование выборки
        fill_top_to, min_group_size = -1, 2

        inp_df = inp_df[['id_left', 'id_right', 'label']]
        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        group_size = inp_df.groupby('id_left').size()
        left_ind_to_use = group_size[group_size >= min_group_size].index.tolist()
        groups = inp_df[inp_df['id_left'].isin(left_ind_to_use)].groupby('id_left')

        np.random.seed(seed)

        out_pairs = []
        for id_left, group in groups:
            ones_ids = group[group.label == 1].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(set(zeroes_ids)).union(set(id_left))
                pad_sample = np.random.choice(list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for doc1, doc2 in self._gen_pairs(ones_ids, zeroes_ids):
                out_pairs.append([id_left, doc1, doc2, 1.0] if np.random.rand() > 0.5 else [id_left, doc2, doc1, 0.0])
            for doc1, doc2 in self._gen_pairs(ones_ids, pad_sample):
                out_pairs.append([id_left, doc1, doc2, 1.0] if np.random.rand() > 0.5 else [id_left, doc2, doc1, 0.0])
            for doc1, doc2 in self._gen_pairs(zeroes_ids, pad_sample):
                out_pairs.append([id_left, doc1, doc2, 1.0] if np.random.rand() > 0.5 else [id_left, doc2, doc1, 0.0])

        return out_pairs

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    def ndcg_k(self, ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
        current_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        if ideal_dcg == 0:
            return 0
        return current_dcg / ideal_dcg

    def _dcg_k(self, ys_true: np.array, ys_pred: np.array, top_k: int) -> float:  
        indices = np.argsort(ys_pred)[::-1]
        ind = min((len(ys_true), top_k))
        ys_true_sorted = ys_true[indices][:ind]
        gain = 0
        for i, y in enumerate(ys_true_sorted, start=1):
            gain += (2 ** y.item() - 1) / math.log2(i + 1)

        return gain

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self.ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def train(self, n_epochs: int):
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()

        # Начальная выборка для тренировки
        train_triplets = self.sample_data_for_train_iter(self.glue_train_df, 0)
        train_dataset = TrainTripletsDataset(train_triplets, self.idx_to_text_mapping_train, self.vocab, self.vocab['OOV'], self.simple_preproc)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.dataloader_bs, num_workers=0, collate_fn=collate_fn, shuffle=True)

        for epoch in range(n_epochs):
            self.model.train()  # Переводим модель в режим тренировки
            epoch_loss = 0.0
            total_batches = 0
            
            # Пересоздаем выборку каждые change_train_loader_ep эпох
            # if (epoch + 1) % self.change_train_loader_ep == 0:
            #     train_triplets = self.sample_data_for_train_iter(self.glue_train_df, epoch)

            #     train_dataset = TrainTripletsDataset(train_triplets, self.idx_to_text_mapping_train, self.vocab, self.vocab['OOV'], self.simple_preproc)
            #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.dataloader_bs, num_workers=0, collate_fn=collate_fn, shuffle=True)

            for batch in train_dataloader:
                inp_left, inp_right, labels = batch
                opt.zero_grad()  # Обнуляем градиенты перед обратным проходом

                # Получаем предсказания модели
                preds = self.model(inp_left, inp_right)

                # Вычисляем потери
                loss = criterion(preds, labels)
                loss.backward()  # Обратное распространение ошибки
                opt.step()  # Обновляем веса

                epoch_loss += loss.item()
                total_batches += 1
            
            # Средняя ошибка за эпоху
            avg_loss = epoch_loss / total_batches
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}')

            # После каждой эпохи проверяем на валидационном наборе
            if (epoch + 1) % (self.change_train_loader_ep / 2) == 0:
                val_ndcg = self.valid(self.model, self.val_dataloader)
                print(f'Epoch {epoch + 1}/{n_epochs}, NDCG: {val_ndcg:.4f}')
        print("Final metrics")
        val_ndcg = self.valid(self.model, self.val_dataloader)
        print(f'Epoch {epoch + 1}/{n_epochs}, NDCG: {val_ndcg:.4f}')
