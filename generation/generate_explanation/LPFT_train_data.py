#!/usr/bin/env python3
import json
import argparse
import random
import numpy as np
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_training_set(file_path):
    """
    Load the training triples and construct a bidirectional graph:
    - forward_graph: subject → [(relation, object)]
    - reverse_graph: object → [(relation, subject)]
    Also collects all triples as a set.
    """
    forward_graph = {}
    reverse_graph = {}
    triples = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            subj, rel, obj = parts

            forward_graph.setdefault(subj, []).append((rel, obj))
            reverse_graph.setdefault(obj, []).append((rel, subj))
            triples.add((subj, rel, obj))

    return (forward_graph, reverse_graph), triples

def find_pattern_pairs(graph_tuple, pattern):
    """
    Given a relation pattern, find all (head, tail) pairs in the graph that match it.
    The pattern can traverse forward and reverse edges.
    """
    forward_graph, reverse_graph = graph_tuple
    pairs = set()

    for start in forward_graph:
        def dfs(node, depth):
            if depth == len(pattern):
                yield node
            else:
                current_rel = pattern[depth]
                if node in forward_graph:
                    for rel, nxt in forward_graph[node]:
                        if rel == current_rel:
                            yield from dfs(nxt, depth + 1)
                if node in reverse_graph:
                    for rel, prev in reverse_graph[node]:
                        if rel == current_rel:
                            yield from dfs(prev, depth + 1)

        for end in dfs(start, 0):
            pairs.add((start, end))

    return pairs

def filter_candidate_paths(candidate_json_file, training_txt_file, topk, m, output_file):
    """
    Filters candidate explanation paths based on pattern confidence scores:
    1. Group paths by their relation pattern.
    2. For each pattern, compute confidence: support(B ∪ r) / support(B).
    3. Rank patterns by score and select top-k patterns.
    4. For each selected pattern, retain up to m example paths.
    """
    graph, triples = load_training_set(training_txt_file)

    with open(candidate_json_file, 'r', encoding='utf-8') as f:
        candidate_data = json.load(f)

    filtered_candidates = {}
    count = 0
    cache_file = 'pattern_cache.pkl'

    # Load cached pattern results to avoid recomputation
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            pattern_cache = pickle.load(f)
        print("Pattern cache loaded.")
    else:
        pattern_cache = {}

    for target, candidate_paths in candidate_data.items():
        print(count)
        count += 1
        target_parts = target.split('\t')
        if len(target_parts) != 3:
            continue
        target_h, target_r, target_t = target_parts

        # Group paths by relation pattern (tuple of relations)
        pattern_groups = {}
        for path in candidate_paths:
            pattern = []
            for edge in path:
                parts = edge.split('\t')
                if len(parts) != 3:
                    continue
                _, rel, _ = parts
                pattern.append(rel)
            if not pattern:
                continue
            pattern_tuple = tuple(pattern)
            pattern_groups.setdefault(pattern_tuple, []).append(path)

        # Compute confidence and normalized support for each pattern
        group_list = []
        max_supp, min_supp = 0, 0
        score_list = []
        i = 0

        for pattern_tuple, paths in pattern_groups.items():
            if pattern_tuple not in pattern_cache:
                pattern_cache[pattern_tuple] = find_pattern_pairs(graph, list(pattern_tuple))
            pairs = pattern_cache[pattern_tuple]
            supp = len(pairs)
            supp_with = sum((h, target_r, t) in triples for (h, t) in pairs)
            confidence = supp_with / supp if supp > 0 else 0.0

            if i == 0:
                min_supp = supp_with
            min_supp = min(min_supp, supp_with)
            max_supp = max(max_supp, supp_with)
            i += 1

            score_list.append((pattern_tuple, confidence, supp_with))

        # Normalize support using log scale
        max_supp = np.log(max_supp + 1)
        min_supp = np.log(min_supp + 1)

        for j, (pattern_tuple, paths) in enumerate(pattern_groups.items()):
            _, conf, supp1 = score_list[j]
            supp = np.log(supp1 + 1)
            supp = (supp - min_supp) / (max_supp - min_supp + 1e-8)
            score = conf + supp
            group_list.append((score, pattern_tuple, paths))

        # Select top-k highest scoring relation patterns
        group_list.sort(key=lambda x: x[0], reverse=True)
        topk_groups = [pattern_tuple for _, pattern_tuple, _ in group_list[:topk]]

        # Select m candidate paths for each pattern
        final_selected_paths = []
        for pattern in topk_groups:
            paths = pattern_groups.get(pattern, [])
            selected = paths[:m]  # optional: random.sample(paths, min(m, len(paths)))
            final_selected_paths.extend(selected)

        filtered_candidates[target] = final_selected_paths

    # Write output to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_candidates, f, indent=2, ensure_ascii=False)

    # Save pattern cache
    with open(cache_file, 'wb') as f:
        pickle.dump(pattern_cache, f)
        print("Pattern cache saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_json", type=str, default="/FB15k-237/FB15k-237_transe_candi_path_1_3_50_20.json")
    parser.add_argument("--train_set", type=str, default="FB15k-237/train.txt")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--output_json", type=str, default="FB15k-237/FB15k-237_transe_candi_path_1_10_all_path.json")
    args = parser.parse_args()

    filter_candidate_paths(args.candidate_json, args.train_set, args.topk, args.m, args.output_json)
