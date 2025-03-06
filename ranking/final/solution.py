from typing import Dict, List, Tuple, Union, Callable
import os
import string
import json

import nltk
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from langdetect import detect
from flask import Flask, request, jsonify


# Загрузка путей из переменных окружения
EMB_PATH_KNRM = os.getenv("EMB_PATH_KNRM")
EMB_PATH_GLOVE = os.getenv("EMB_PATH_GLOVE")
VOCAB_PATH = os.getenv("VOCAB_PATH")
MLP_PATH = os.getenv("MLP_PATH")

# Инициализация Flask-сервера
app = Flask(__name__)

# Глобальные переменные
faiss_index = None
vocab = {}
knrm_model = None

def load_glove_embeddings(glove_path):
    """Загружает GloVe-эмбеддинги в словарь."""
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def load_vocab(vocab_path):
    """Загружает словарь индексов KNRM."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)

def hadle_punctuation(inp_str: str) -> str:
    for p in string.punctuation:
        inp_str = inp_str.replace(p, ' ')
    return inp_str

def simple_preproc(inp_str: str) -> List[str]:
    processed_str = hadle_punctuation(inp_str)
    return nltk.word_tokenize(processed_str.lower())

def text2token_ids(texts: List[str]) -> torch.LongTensor:
    tokenized = []
    for text in texts:
        tokenized_text = simple_preproc(text)
        token_idxs = [vocab.get(i, vocab["OOV"]) for i in tokenized_text]
        tokenized.append(token_idxs)
    max_len = max(len(elem) for elem in tokenized)
    tokenized = [elem + [0] * (max_len - len(elem)) for elem in tokenized]
    tokenized = torch.LongTensor(tokenized)    
    return tokenized

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-(x-self.mu)**2 / (2 * self.sigma**2))


class KNRM(torch.nn.Module):
    def __init__(self, embedding_path: str, pretrained_mlp_path: str, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.load(embedding_path)["weight"],
            freeze=True,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(torch.load(pretrained_mlp_path))
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out

@app.route("/ping", methods=["GET"])
def ping():
    """Эндпоинт для проверки готовности сервиса."""
    if knrm_model is None:
        return jsonify({"status": "not ready!"})
    return jsonify({"status": "ok"})

@app.route("/update_index", methods=["POST"])
def update_index():
    """Создаёт FAISS-индекс по загруженным вопросам."""
    global faiss_index, index_is_ready, documents
    data = request.get_json()
    documents = data.get("documents", {})
    # nonlocal documents
    if not documents:
        return jsonify({"status": "error", "message": "No documents received"})
    
    oov_val = vocab.get("OOV", 0)
    emb_layer = knrm_model['weight']
    
    idxs, embeddings = [], []
    for doc_id, text in documents.items():
        idxs.append(int(doc_id))
        token_ids = [vocab.get(w, oov_val) for w in simple_preproc(text)]
        text_embedding = emb_layer[token_ids].mean(dim=0).numpy()
        embeddings.append(text_embedding)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index = faiss.IndexIDMap(faiss_index)
    faiss_index.add_with_ids(embeddings, np.array(idxs))
    
    index_is_ready = True
    return jsonify({"status": "ok", "index_size": faiss_index.ntotal})

@app.route("/query", methods=["POST"])
def get_ranked_candidates():
    """
    Принимает POST-запрос. Сначала происходит фильтрация запроса по языку - только английский.
    Затем происходит поиск вопросов-кандидатов с помощью FAISS (по схожести векторов).
    Эти кандидаты реранжируются KNRM-моделью, после чего до 10 кандидатов выдаются в качестве ответа.
    """
    global knrm_model, faiss_index, vocab
    if not index_is_ready:
        return jsonify({"status": 'FAISS is not initialized!'})
    data = request.get_json()
    queries = data.get("queries", [])
    lang_checks, suggestions = [], []
    emb_layer = knrm_model['weight']
    topk = 10
    # nonlocal documents
    for query in queries:
        if detect(query) != "en":
            lang_checks.append(False)
            suggestions.append(None)
        else:
            lang_checks.append(True)

            query_tokenized = simple_preproc(query)
            query_token_ids = [vocab.get(tok, vocab["OOV"]) for tok in query_tokenized]
            query_embedding = emb_layer[query_token_ids].mean(dim=0).reshape(1, -1).numpy()

            inds = faiss_index.search(query_embedding, k=100)
            candidates = [(str(i), documents[str(i)]) for i in inds[0] if i != -1]
            outputs = knrm_model({
                "query": text2token_ids([query * len(candidates)]),
                "document": text2token_ids([cand[1] for cand in candidates])
            })
            topk_candidate_ids = outputs.reshape(-1).argsort(descending=True)[:topk]
            topk_candidates = [candidates[i] for i in topk_candidate_ids.tolist()]
            suggestions.append(topk_candidates)
    return jsonify(lang_check=lang_checks, suggestions=suggestions)


if __name__ == "__main__":
    glove_embeddings = load_glove_embeddings(EMB_PATH_GLOVE)
    vocab = load_vocab(VOCAB_PATH)
    
    knrm_model = KNRM(
      emb_path=EMB_PATH_KNRM,
      mlp_path=MLP_PATH,
    )

    app.run(host="0.0.0.0", port=11000)
