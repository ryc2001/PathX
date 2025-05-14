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

def replace_triples_with_semantics(input_dict):
    replaced_dict = {}

    for triple, paths in input_dict.items():
        head, relation, tail = triple.split("\t")
        head_semantic = synset_id_to_name(head)
        tail_semantic = synset_id_to_name(tail)

        replaced_paths = []
        for path in paths:
            replaced_path = []
            for step in path:
                step_head, step_relation, step_tail = step.split("\t")
                step_head_semantic = synset_id_to_name(step_head)
                step_tail_semantic = synset_id_to_name(step_tail)
                replaced_path.append(f"{step_head_semantic}\t{step_relation}\t{step_tail_semantic}")
            replaced_paths.append(replaced_path)

        replaced_dict[f"{head_semantic}\t{relation}\t{tail_semantic}"] = replaced_paths

    return replaced_dict

def save_dict_to_json(data, output_file):

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":

    with open("baseline/GR/complex_wn_new.json", "r") as f:
        input_data = json.load(f)
    replaced_data = replace_triples_with_semantics(input_data)

    output_file = "baseline/GR/wn_complex_sem.json"
    save_dict_to_json(replaced_data, output_file)
