import heapq

import numpy as np
import torch
from sklearn.neighbors import KDTree


def repeated_lemma_bool(word, path, lemmatizer):
    # Check noun, verb, and adjective lemmas. Different senses of the same word are excluded
    for i in path:
        for pos_tag in ["n", "v", "a"]:
            if lemmatizer.lemmatize(
                word.split("_")[0], pos_tag
            ) == lemmatizer.lemmatize(i.split("_")[0], pos_tag) and not (
                word.split("_")[0] == i.split("_")[0]
                and word.split("_")[1] != i.split("_")[1]
            ):
                return True
    return False


def morph_a_star_generator(
    word_start,
    word_end,
    model,
    lemmatizer,
    k_neighbors,
    max_neighbor_distance,
    k_cutoff,
    temperature,
    weights,
    max_expansions,
    allow_immediate_reach,
    centroids_path,
    centroid_metadata,
    device,
    yield_interval=None,
    kdtree=None,
):

    d = centroid_metadata["dimension"]
    centroid_count = centroid_metadata["centroid_count"]
    centroids_memmap = np.memmap(
        centroids_path, dtype=np.float32, mode="r", shape=(centroid_count, d)
    )
    if kdtree is None:
        kdtree = KDTree(centroids_memmap)
    word_to_centroid_indices = centroid_metadata["word_to_centroid_indices"]

    index_to_word = {}
    for word, indices in word_to_centroid_indices.items():
        for i, idx in enumerate(indices):
            index_to_word[idx] = word + f"_{i}"

    idx_a = word_to_centroid_indices.get(f"{word_start}")
    idx_b = word_to_centroid_indices.get(f"{word_end}")
    if idx_a is None:
        print(
            f'The word "{word_start}" is not in the vocabulary. Please try another word.'
        )
        yield {"status": 0}
    if idx_b is None:
        print(
            f'The word "{word_end}" is not in the vocabulary. Please try another word.'
        )
        yield {"status": 0}
    a_senses = centroids_memmap[idx_a]
    b_senses = centroids_memmap[idx_b]

    # Find the best pair of senses to initialize the search
    best_pair = min(
        [(i, j) for i in range(len(a_senses)) for j in range(len(b_senses))],
        key=lambda ij: np.linalg.norm(a_senses[ij[0]] - b_senses[ij[1]]),
    )

    start = f"{word_start}_{best_pair[0]}"
    end = f"{word_end}_{best_pair[1]}"
    start_vec = a_senses[best_pair[0]]
    end_vec = b_senses[best_pair[1]]

    end_tensor = torch.from_numpy(end_vec).float().to(device)
    start_centroid_idx = word_to_centroid_indices[f"{word_start}"][best_pair[0]]
    end_centroid_idx = word_to_centroid_indices[f"{word_end}"][best_pair[1]]

    start_h = np.linalg.norm(start_vec - end_vec)

    heap = []
    heapq.heappush(heap, (0, start, [start], 0))  # (f, node, path, g)

    expanded_word_senses = set()
    expanded_indices = set()
    path = []

    expansions = 0

    while heap and expansions < max_expansions:
        _, current, path, g = heapq.heappop(heap)

        # Accept exact senses of the target word
        if path[-1] == end:
            centroid_path = (
                [start_centroid_idx]
                + [
                    word_to_centroid_indices[node.split("_")[0]][
                        int(node.split("_")[1])
                    ]
                    for node in path[1:-1]
                ]
                + [end_centroid_idx]
            )
            result = {
                "status": 2,
                "word_sense_path": path,
                "centroid_path": centroid_path,
                "expansion_count": expansions,
                "expanded_word_senses": list(expanded_word_senses),
                "expanded_indices": list(expanded_indices),
            }
            yield result
            return

        # Skip visited nodes
        if current in expanded_word_senses:
            continue
        expanded_word_senses.add(current)
        expanded_indices.add(
            word_to_centroid_indices[current.split("_")[0]][int(current.split("_")[1])]
        )

        current_vec = centroids_memmap[
            word_to_centroid_indices[current.split("_")[0]][int(current.split("_")[1])]
        ]
        zt_tensor = torch.from_numpy(current_vec.copy()).float().to(device)

        # Flow guidance
        with torch.inference_mode():
            # Estimate progress by the distance to the target centroid
            current_h = np.linalg.norm(current_vec - end_vec)

            t_val = np.clip(1.0 - (current_h / (start_h + 1e-8)), 0, 1)
            t = torch.tensor([t_val]).float().to(device)

            inp = torch.cat([zt_tensor, end_tensor, t], dim=-1)
            d = model(inp.unsqueeze(0)).squeeze(0).cpu().numpy()
            d_norm = d / (np.linalg.norm(d) + 1e-8)

        # Find k nearest neighbors, subject to a maximum neighbor distance
        dists, idxs = kdtree.query(current_vec.reshape(1, -1), k=k_neighbors)
        filtered = [
            (d, i) for d, i in zip(dists[0], idxs[0]) if d <= max_neighbor_distance
        ]
        if not filtered:
            continue

        if temperature > 0:
            dists, idxs = zip(*filtered)

            dists = np.asarray(dists).reshape(-1)
            idxs = np.asarray(idxs).reshape(-1)

            scores = -dists  # closer = higher score

            probs = np.exp(scores / temperature)
            probs /= probs.sum()

            k_sample = min(len(idxs), k_cutoff)
            chosen = np.random.choice(len(idxs), size=k_sample, replace=False, p=probs)

            candidates = [index_to_word[idxs[i]] for i in chosen]
        else:
            candidates = [index_to_word[idx] for d, idx in filtered]

        for nxt in candidates:
            if nxt in expanded_word_senses:
                continue

            nxt_vec = centroids_memmap[
                word_to_centroid_indices[nxt.split("_")[0]][int(nxt.split("_")[1])]
            ]

            step_cost = np.linalg.norm(nxt_vec - current_vec)

            move = nxt_vec - current_vec
            move_norm = move / (np.linalg.norm(move) + 1e-8)

            flow_alignment = np.dot(move_norm, d_norm)

            # Prune unmatched directions
            if weights[2] > 0 and flow_alignment < 0:
                continue
            clean_name = nxt.split("_")[0]

            # Whether to avoid immediate reach if the two words are too similar
            if clean_name == word_end and not allow_immediate_reach and len(path) < 2:
                continue

            # Will not punish semantic shift by excluding same words but different senses
            lemma_bool = repeated_lemma_bool(nxt, path + [end], lemmatizer)

            flow_cost = 1.0 - flow_alignment  # Higher alignment means lower cost
            cost = (
                weights[0] * step_cost
                + weights[1] * lemma_bool
                + weights[2] * flow_cost
            )

            g_new = g + cost

            dist_to_goal = np.linalg.norm(nxt_vec - end_vec)
            h_base = weights[0] * dist_to_goal
            h = h_base * (1.0 + weights[2] * flow_cost)

            f_new = g_new + h

            heapq.heappush(heap, (f_new, nxt, path + [nxt], g_new))

        expansions += 1

        if (
            yield_interval
            and yield_interval > 0
            and (expansions == 1 or expansions % yield_interval == 0)
        ):
            centroid_path = [start_centroid_idx] + [
                word_to_centroid_indices[node.split("_")[0]][int(node.split("_")[1])]
                for node in path
            ]
            yield {
                "status": 1,
                "word_sense_path": path,
                "centroid_path": centroid_path,
                "expansion_count": expansions,
                "expanded_word_senses": list(expanded_word_senses),
                "expanded_indices": list(expanded_indices),
            }

    print(
        f'No path found from the word "{word_start}" to "{word_end}" within A* expansion limit.'
    )
    yield {"status": 0}
