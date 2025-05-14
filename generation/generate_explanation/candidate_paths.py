#!/usr/bin/env python3
import json
import argparse
import random

def load_training_set(file_path):
    graph = {}
    triples = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            if subj not in graph:
                graph[subj] = []
            graph[subj].append((rel, obj))
            triples.add((subj, rel, obj))
    return graph, triples

def find_pattern_pairs(graph, pattern):
    pairs = set()
    for start in graph:
        def dfs(node, depth):
            if depth == len(pattern):
                yield node
            else:
                if node in graph:
                    for rel, nxt in graph[node]:
                        if rel == pattern[depth]:
                            yield from dfs(nxt, depth + 1)
        for end in dfs(start, 0):
            pairs.add((start, end))
    return pairs

def filter_candidate_paths(candidate_json_file, training_txt_file, m, output_file):
    graph, triples = load_training_set(training_txt_file)
    with open(candidate_json_file, 'r', encoding='utf-8') as f:
        candidate_data = json.load(f)
    filtered_candidates = {}
    count = 0
    for target, candidate_paths in candidate_data.items():
        print(count)
        count += 1
        target_parts = target.split('\t')
        if len(target_parts) != 3:
            continue
        target_h, target_r, target_t = target_parts
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
            if pattern_tuple not in pattern_groups:
                pattern_groups[pattern_tuple] = []
            pattern_groups[pattern_tuple].append(path)
        group_list = []
        for pattern_tuple, paths in pattern_groups.items():
            pairs = find_pattern_pairs(graph, list(pattern_tuple))
            supp = len(pairs)
            supp_with = 0
            for (h_candidate, t_candidate) in pairs:
                if (h_candidate, target_r, t_candidate) in triples:
                    supp_with += 1
            confidence = supp_with / supp if supp > 0 else 0.0
            group_list.append((confidence, pattern_tuple, paths))
        group_list.sort(key=lambda x: x[0], reverse=True)
        topk_groups = group_list
        final_patterns = topk_groups
        final_selected_paths = []
        for pattern in final_patterns:
            paths = pattern_groups.get(pattern, [])
            random.shuffle(paths)
            unique_paths = []
            seen_h = set()
            for path in paths:
                path_tuple = tuple(path)
                if path_tuple[0] not in seen_h:
                    unique_paths.append(path)
                    seen_h.add(path_tuple[0])
            def calculate_path_degree(path):
                degree_sum = 0
                if len(path) == 2:
                    parts = path[0].split('\t')
                    if len(parts) == 3:
                        _, _, middle_entity = parts
                        if middle_entity in graph:
                            degree_sum += len(graph[middle_entity])
                elif len(path) == 3:
                    parts1 = path[0].split('\t')
                    if len(parts1) == 3:
                        _, _, middle_entity1 = parts1
                        if middle_entity1 in graph:
                            degree_sum += len(graph[middle_entity1])
                    parts2 = path[1].split('\t')
                    if len(parts2) == 3:
                        _, _, middle_entity2 = parts2
                        if middle_entity2 in graph:
                            degree_sum += len(graph[middle_entity2])
                return degree_sum
            path_degrees = [(path, calculate_path_degree(path)) for path in unique_paths]
            path_degrees.sort(key=lambda x: x[1], reverse=True)
            top_m_paths = [path for path, _ in path_degrees[:m]]
            final_selected_paths.extend(top_m_paths)
        filtered_candidates[target] = final_selected_paths
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_candidates, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_json", type=str, default="WN18RR_candi_path_1_complex.json")
    parser.add_argument("--train_set", type=str, default="FB15k-237/train.txt")
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--output_json", type=str, default="/WN18RR/WN18RR_complex_candi_path_xin2.json")
    args = parser.parse_args()
    filter_candidate_paths(args.candidate_json, args.train_set, args.m, args.output_json)
