import csv
import json
import sys
from collections import defaultdict

def sel_influential_rels(csv_file, json_file, target_relation, ratio):
    with open(json_file, 'r') as f:
        relation_map = json.load(f)
    target_relation_index = target_relation

    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        csv_matrix = [list(map(float, row)) for row in csv_reader]

    n = len(csv_matrix)
    target_row = csv_matrix[target_relation_index]
    half_n = n // 2
    sel_num = int(ratio * (n // 2))

    top_indices_and_values = sorted(
        [(i, target_row[i]) for i in range(half_n)],
        key=lambda x: x[1],
        reverse=True
    )[:sel_num]

    top_relations = [
        index
        for index, value in top_indices_and_values
        for relation, idx in relation_map.items()
        if idx - 1 == index
    ]
    return top_relations

def extract_influential_subgraph(triples, head_entity, tail_entity, influential_rels):
    outgoing_edges = defaultdict(list)
    incoming_edges = defaultdict(list)
    for h, r, t in triples:
        outgoing_edges[h].append((r, t))
        incoming_edges[t].append((r, h))

    def get_two_hop_triples(entity):
        one_hop_triples = []
        two_hop_triples = []
        if entity in outgoing_edges:
            for r1, neighbor in outgoing_edges[entity]:
                one_hop_triples.append((entity, r1, neighbor))
        if entity in incoming_edges:
            for r1, neighbor in incoming_edges[entity]:
                one_hop_triples.append((neighbor, r1, entity))
        return one_hop_triples, two_hop_triples

    head_one_hop, head_two_hop = get_two_hop_triples(head_entity)
    tail_one_hop, tail_two_hop = get_two_hop_triples(tail_entity)

    all_triples = set(head_one_hop + tail_one_hop)
    final_triples = set()
    for h, r, t in all_triples:
        if r in influential_rels:
            final_triples.add((h, r, t))
    return final_triples

def find_paths(subgraph, head_entity, tail_entity):
    one_hop_triples = []
    for triple in subgraph:
        h, r, t = triple
        if (h == head_entity and t == tail_entity) or (h == tail_entity and t == head_entity):
            one_hop_triples.append((h, r, t))
        if h == head_entity or t == head_entity or h == tail_entity or t == tail_entity:
            one_hop_triples.append((h, r, t))

    path_triples = set()
    path_triples = path_triples.union(set(one_hop_triples))
    return path_triples
