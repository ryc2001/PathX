import json
from nltk.corpus import wordnet as wn

def synset_id_to_name(synset_id):
    try:
        synset_offset = int(synset_id)
        for pos in ['n', 'v', 'a', 'r']:
            try:
                synset = wn.synset_from_pos_and_offset(pos, synset_offset)
                return f"{synset.name()}: {synset.definition()}"
            except:
                continue
        return "Not Found"
    except ValueError:
        return "Invalid ID"

def replace_triples_with_semantics(input_lines, id2name):
    i = 0
    data = []
    new_dict = {}
    while i <= len(input_lines) - 3:
        fact_line = input_lines[i]
        rules_line = input_lines[i + 1]
        # sample to explain
        head, relation, tail = tuple(fact_line.strip().split(";"))
        head_semantic = id2name.get(head, head)
        tail_semantic = id2name.get(tail, tail)
        target = f'{head_semantic}\t{relation}\t{tail_semantic}'
        exp = []
        if rules_line.strip() != "":
            rules_with_relevance = []
            rule_relevance_inputs = rules_line.strip().split(",")
            count = 0
            for tp in rule_relevance_inputs:
                rule = tp.split(":")[0]
                rule_bits = rule.split(";")
                cur_head_name = rule_bits[0]
                cur_rel_name = rule_bits[1]
                cur_tail_name = rule_bits[2]
                exp.append([f'{id2name.get(cur_head_name, cur_head_name)}\t{cur_rel_name}\t{id2name.get(cur_tail_name, cur_tail_name)}'])
                count += 1
                # if count == 5:
                #     break
        i += 3
        new_dict[target] = exp
    return new_dict

def load_mapping(mapping_file):
    id2name = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mid, name = parts
                id2name[mid] = name
    return id2name

def save_dict_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 示例用法
if __name__ == "__main__":
    mapping_path = 'baseline/ours/FB15k_mid2name.txt'
    mapping = load_mapping(mapping_path)
    with open("/data/zhaotianzhe/ryc/KGEvaluator/evaluation/baseline/kelpie/output_None_complex_fb.txt", "r") as f:
        input_data = f.readlines()
    replaced_data = replace_triples_with_semantics(input_data, mapping)

    output_file = "ceshi-kelpie.txt"
    save_dict_to_json(replaced_data, output_file)