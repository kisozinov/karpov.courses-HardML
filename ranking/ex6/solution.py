from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List
import heapq

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return np.linalg.norm(documents - pointA, axis=1, keepdims=True)

def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    sw_edges = defaultdict(list)
    for i in range(data.shape[0]):
        if use_sampling:
            sample_size = int(data.shape[0] * sampling_share)
            sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
            distances = dist_f(data[i], data[sample_indices])
            indices = sample_indices
        else:
            distances = dist_f(data[i], data)
            indices = np.arange(data.shape[0])

        sorted_indices = indices[np.argsort(distances, axis=0).flatten()]
        sorted_indices = sorted_indices[sorted_indices != i]

        candidates_long_edge = sorted_indices[-num_candidates_for_choice_long:]
        long_edges = np.random.choice(candidates_long_edge, num_edges_long, replace=False)
        sw_edges[i].extend(long_edges)

        candidates_short_edge = sorted_indices[:num_candidates_for_choice_short]
        short_edges = np.random.choice(candidates_short_edge, num_edges_short, replace=False)
        sw_edges[i].extend(short_edges)
        
    return sw_edges
        

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:

    visited = set()
    heap = []  # Очередь ближайших (мин-куча)
    num_start = max(search_k, num_start_points)
    # start_nodes = np.random.choice(list(graph_edges.keys()), num_start, replace=False)
    best_candidates = []

    while len(best_candidates) < search_k:
        node = np.random.randint(0, len(all_documents), 1)[0]
        if node not in visited:
            min_dist = dist_f(query_point, all_documents[node].reshape(1, -1))[0, 0]
            heapq.heappush(heap, (min_dist, node))
            visited.add(node)
            stop_condition = False
            while not stop_condition:
            # select best neighbour
                edges = graph_edges[node]
                dists = dist_f(query_point, all_documents[edges])
                cur_min_dist, edge_min = dists.min(), edges[dists.flatten().argmin()]
                if cur_min_dist <= min_dist:
                    min_dist = cur_min_dist
                    node = edge_min
                else:
                    stop_condition = True
            heapq.heappush(heap, (min_dist, node))
            best_candidates.append((min_dist, node))

    return np.array([node for _, node in best_candidates[:search_k]])
            



    # def add_start_nodes(n):
    #     """Добавляет `n` стартовых вершин в очередь"""
    #     start_nodes = np.random.choice(list(graph_edges.keys()), n, replace=False)
    #     for node in start_nodes:
    #         if node not in visited:
    #             dist = dist_f(query_point, all_documents[node].reshape(1, -1))[0, 0]
    #             heapq.heappush(heap, (dist, node))
    #             visited.add(node)

    # # 1. Добавляем начальные стартовые точки
    # add_start_nodes(num_start_points)

    # best_candidates = []
    # while heap and len(best_candidates) < search_k:
    #     dist, current = heapq.heappop(heap)  # Берем ближайшего кандидата
    #     best_candidates.append((dist, current))

    #     # 2. Добавляем его соседей
    #     for neighbor in graph_edges[current]:
    #         if neighbor not in visited:
    #             visited.add(neighbor)
    #             neighbor_dist = dist_f(query_point, all_documents[neighbor].reshape(1, -1))[0, 0]
    #             heapq.heappush(heap, (neighbor_dist, neighbor))

    # # 3. Если точек не хватает, увеличиваем стартовые вершины
    # while len(best_candidates) < search_k:
    #     add_start_nodes(2)  # Добавляем еще 2 стартовые точки
    #     while heap and len(best_candidates) < search_k:
    #         dist, current = heapq.heappop(heap)
    #         best_candidates.append((dist, current))

    # # 4. Сортируем и возвращаем `search_k` лучших
    # best_candidates.sort()
    # return np.array([node for _, node in best_candidates[:search_k]])