"""Verbatim algorithm definitions from corrected seed-20261015 v1.2 evaluation notebook,
code cell 2. No algorithmic changes; imports and global constants added as module context.
Do not alter if reproducing reported classical comparisons.
"""
import numpy as np

ACO_ANTS, ACO_ITERATIONS, ACO_ALPHA, ACO_BETA, ACO_RHO, ACO_Q = 15, 30, 1.0, 3.0, 0.1, 1.0

def tour_length_np(coords, tour):
    coords = np.asarray(coords, dtype=np.float64)
    tour = np.asarray(tour, dtype=np.int64)
    ordered = coords[tour]
    return float(np.linalg.norm(ordered[1:] - ordered[:-1], axis=1).sum()
                 + np.linalg.norm(ordered[0] - ordered[-1]))


def distance_matrix(coords):
    delta = np.asarray(coords)[:, None, :] - np.asarray(coords)[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def nearest_neighbor(coords, start=0):
    coords = np.asarray(coords, dtype=np.float64); n = len(coords); d = distance_matrix(coords)
    tour = [int(start)]; unvisited = set(range(n)); unvisited.remove(int(start))
    while unvisited:
        current = tour[-1]
        nxt = min(unvisited, key=lambda j: (d[current, j], j))
        tour.append(nxt); unvisited.remove(nxt)
    return np.asarray(tour, dtype=np.int64)


def two_opt(coords, initial_tour, max_passes=100):
    coords = np.asarray(coords, dtype=np.float64); d = distance_matrix(coords)
    tour = np.asarray(initial_tour, dtype=np.int64).copy(); n = len(tour)
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: continue
                a, b = tour[i], tour[(i + 1) % n]; c, e = tour[j], tour[(j + 1) % n]
                if d[a, b] + d[c, e] > d[a, c] + d[b, e] + 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True; break
            if improved: break
        if not improved: break
    return tour


def ant_colony_tsp(coords, seed, n_ants=ACO_ANTS, n_iter=ACO_ITERATIONS,
                   alpha=ACO_ALPHA, beta=ACO_BETA, rho=ACO_RHO, q=ACO_Q):
    coords = np.asarray(coords, dtype=np.float64); n = len(coords); d = distance_matrix(coords)
    heuristic = 1.0 / np.maximum(d, 1e-12); np.fill_diagonal(heuristic, 0.0)
    pheromone = np.ones((n, n), dtype=np.float64); rng = np.random.default_rng(seed)
    best_tour, best_length = None, float('inf')
    for _ in range(n_iter):
        iteration_tours, iteration_lengths = [], []
        for _ant in range(n_ants):
            start = int(rng.integers(0, n)); tour = [start]
            unvisited = np.ones(n, dtype=bool); unvisited[start] = False
            while unvisited.any():
                current = tour[-1]; candidates = np.flatnonzero(unvisited)
                weights = np.power(pheromone[current, candidates], alpha) * np.power(heuristic[current, candidates], beta)
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                if weights.sum() <= 0: nxt = int(rng.choice(candidates))
                else: nxt = int(rng.choice(candidates, p=weights / weights.sum()))
                tour.append(nxt); unvisited[nxt] = False
            tour = np.asarray(tour, dtype=np.int64); length = tour_length_np(coords, tour)
            iteration_tours.append(tour); iteration_lengths.append(length)
            if length < best_length: best_tour, best_length = tour.copy(), length
        pheromone *= (1.0 - rho)
        for tour, length in zip(iteration_tours, iteration_lengths):
            deposit = q / max(length, 1e-12)
            for k in range(n):
                a, b = int(tour[k]), int(tour[(k + 1) % n])
                pheromone[a, b] += deposit; pheromone[b, a] += deposit
    return best_tour

