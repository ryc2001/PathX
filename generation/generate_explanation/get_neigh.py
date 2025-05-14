import csv
import json
import sys
from collections import defaultdict

def sel_influential_rels(csv_file, json_file, target_relation, ratio):
    # Step 1: 读取 JSON 文件，创建关系与数字的映射
    with open(json_file, 'r') as f:
        relation_map = json.load(f)  # JSON 格式：{"/location/country/form_of_government": 1, ...}
    target_relation_index = target_relation

    # Step 2: 读取 CSV 文件
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        csv_matrix = [list(map(float, row)) for row in csv_reader]  # 转换为浮点数矩阵

    # Step 3: 确定矩阵的行数和列数
    n = len(csv_matrix)
    # Step 4: 找到对应行，并在前 N/2 列中搜索
    target_row = csv_matrix[target_relation_index]  # 获取目标关系对应的行
    half_n = n // 2
    sel_num = int(ratio * (n // 2))

    # 在前 N/2 列中找到最大的 N/4 个数字及其列索引
    top_indices_and_values = sorted(
        [(i, target_row[i]) for i in range(half_n)],
        key=lambda x: x[1],  # 按值排序
        reverse=True  # 从大到小
    )[:sel_num]

    # Step 5: 根据这些索引找到对应关系的名称
    top_relations = [
       index
        for index, value in top_indices_and_values
        for relation, idx in relation_map.items()
        if idx - 1 == index
    ]
    return top_relations

def extract_influential_subgraph(triples, head_entity, tail_entity, influential_rels):
    """
    从知识图谱中提取以头实体和尾实体为中心的二跳子图，并将其保存到文件中。

    参数：
    - triples_file: 知识图谱三元组的文件路径（JSON 格式或 TSV 文件）
    - head_entity: 给定的头实体
    - tail_entity: 给定的尾实体
    - output_file: 保存二跳子图的文件路径
    """
    outgoing_edges = defaultdict(list)  # 出边：从实体指向其他实体
    incoming_edges = defaultdict(list)  # 入边：从其他实体指向该实体
    for h, r, t in triples:
        outgoing_edges[h].append((r, t))  # 出边
        incoming_edges[t].append((r, h))  # 入边

    # Step 3: 找到头实体和尾实体的二跳邻居
    def get_two_hop_triples(entity):
        """
        获取一个实体的一跳和二跳三元组（包括入边和出边）。
        """
        one_hop_triples = []  # 一跳三元组
        two_hop_triples = []  # 二跳三元组
        # 一跳邻居：出边
        if entity in outgoing_edges:
            for r1, neighbor in outgoing_edges[entity]:
                one_hop_triples.append((entity, r1, neighbor))
                # 二跳邻居：通过出边的目标实体再找出边
                # if neighbor in outgoing_edges:
                #     for r2, second_neighbor in outgoing_edges[neighbor]:
                #         two_hop_triples.append((neighbor, r2, second_neighbor))
                # if neighbor in incoming_edges:
                #     for r2, second_neighbor in incoming_edges[neighbor]:
                #         two_hop_triples.append((second_neighbor, r2, neighbor ))

        # 一跳邻居：入边
        if entity in incoming_edges:
            for r1, neighbor in incoming_edges[entity]:
                one_hop_triples.append((neighbor, r1, entity))
                # 二跳邻居：通过入边的来源实体再找入边
                # if neighbor in incoming_edges:
                #     for r2, second_neighbor in incoming_edges[neighbor]:
                #         two_hop_triples.append((second_neighbor, r2, neighbor))
                # if neighbor in outgoing_edges:
                #     for r2, second_neighbor in outgoing_edges[neighbor]:
                #         two_hop_triples.append((neighbor, r2, second_neighbor))

        return one_hop_triples, two_hop_triples

    # 提取头实体和尾实体的二跳子图
    head_one_hop, head_two_hop = get_two_hop_triples(head_entity)
    tail_one_hop, tail_two_hop = get_two_hop_triples(tail_entity)

    # Step 4: 合并头实体和尾实体的子图（去重）
    # all_triples = set(head_one_hop + head_two_hop + tail_one_hop + tail_two_hop )
    all_triples = set(head_one_hop + tail_one_hop )
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
            one_hop_triples.append((h,r,t))  # 如果起点到终点直接相连
        if h == head_entity or t == head_entity or h == tail_entity or t == tail_entity:
            one_hop_triples.append((h, r, t))  # 如果起点到终点直接相连
    # two_hop_triples = []
    # for triple_1 in subgraph:
    #     h1, r1, t1 = triple_1
    #     if h1 == head_entity :  # 第一跳从目标头实体出发
    #         for triple_2 in subgraph:
    #             h2, r2, t2 = triple_2
    #             if (t1 == h2 and t2 == tail_entity) or (t1 == t2 and h2 == tail_entity):  # 第二跳连接到目标尾实体
    #                 two_hop_triples.append((h1,r1,t1))  # 第一跳三元组
    #                 two_hop_triples.append((h2,r2,t2))  # 第二跳三元组
    #     if t1 == head_entity :  # 第一跳从目标头实体出发
    #         for triple_2 in subgraph:
    #             h2, r2, t2 = triple_2
    #             if (h1 == h2 and t2 == tail_entity) or (h1 == t2 and h2 == tail_entity):  # 第二跳连接到目标尾实体
    #                 two_hop_triples.append((h1,r1,t1))  # 第一跳三元组
    #                 two_hop_triples.append((h2,r2,t2))  # 第二跳三元组
    path_triples = set()
    path_triples = path_triples.union(set(one_hop_triples))
    # path_triples = path_triples.union(set(two_hop_triples))

    return path_triples