import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--result_path", type=str)
args = parser.parse_args()

def output(filepath):
    fact_to_explain_2_details = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")

            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _explanation_bits = bits[3:-4]
            assert len(_explanation_bits) % 3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                   _explanation_bits[i + 1], \
                                                                                   _explanation_bits[i + 2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i += 3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            fact_to_explain_2_details[_fact_to_explain] = (
                _explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)

    return fact_to_explain_2_details

def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count / float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return round(reciprocal_rank_sum / float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum / float(len(ranks)), 3)


pathX_ROOT = 'generation/'
END_TO_END_EXPERIMENT_ROOT = 'generation/'

systems = ["pathX"]
models = ["TransE", "ComplEx", "ConvE"]
datasets = ["FB15k237", "WN18RR"]
save = args.save

output_data = []
row_labels = []
for model in models:
    for system in systems:
        row_labels.append(f'{model} {system}')
        new_data_row = []
        for dataset in datasets:
            original_ranks = []
            new_ranks = []
            fact_2_pathX_explanations = output(args.result_path)
            for fact_to_explain in fact_2_pathX_explanations:
                pathX_expl, _, _, pathX_original_tail_rank, pathX_new_tail_rank = fact_2_pathX_explanations[
                        fact_to_explain]

                original_ranks.append(pathX_original_tail_rank)
                new_ranks.append(pathX_new_tail_rank)
            original_mrr, original_h1 = mrr(original_ranks), hits_at_k(original_ranks, 1)
            pathX_mrr, pathX_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
            mrr_difference, h1_difference = round(pathX_mrr - original_mrr, 3), round(pathX_h1 - original_h1, 3)
            print(system,model,dataset,pathX_mrr)
            print(system,model,dataset,original_mrr)
            print(system,model,dataset,pathX_h1)
            print(system,model,dataset,original_h1)
            mrr_difference_str = "+" + str(mrr_difference) if mrr_difference > 0 else str(mrr_difference)
            h1_difference_str = "+" + str(h1_difference) if h1_difference > 0 else str(h1_difference)

            new_data_row.append(h1_difference_str)
            new_data_row.append(mrr_difference_str)
        output_data.append(new_data_row)


