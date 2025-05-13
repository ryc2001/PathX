import re
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import copy
import random
import numpy
import torch
import json
from dataset import ALL_DATASET_NAMES, Dataset, MANY_TO_ONE, ONE_TO_ONE
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from link_prediction.models.transe import TransE
from link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, REGULARIZER_WEIGHT

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--dataset",
                    type=str,
                    choices=ALL_DATASET_NAMES)

parser.add_argument("--max_epochs",
                    type=int,
                    default=1000,
                    help="Number of epochs.")

parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size.")

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.0005,
                    help="Learning rate.")

parser.add_argument("--dimension",
                    type=int,
                    default=200,
                    help="Embedding dimensionality.")

parser.add_argument("--margin",
                    type=int,
                    default=5,
                    help="Margin for pairwise ranking loss.")

parser.add_argument("--negative_samples_ratio",
                    type=int,
                    default=3,
                    help="Number of negative samples for each positive sample.")

parser.add_argument("--regularizer_weight",
                    type=float,
                    default=0.0,
                    help="Weight for L2 regularization.")

parser.add_argument("--model_path",
                    type=str,
                    help="Path where to find the model to explain")
parser.add_argument("--result_path",
                    type=str,
                    help="Path where to find the model to explain")


args = parser.parse_args()

seed = 42
torch.backends.cudnn.deterministic = True
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())

########## LOAD DATASET
# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

# get the ids of the elements of the fact to explain and the perspective entity
hyperparameters = {DIMENSION: args.dimension,
                   MARGIN: args.margin,
                   NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
                   REGULARIZER_WEIGHT: args.regularizer_weight,
                   BATCH_SIZE: args.batch_size,
                   LEARNING_RATE: args.learning_rate,
                   EPOCHS: args.max_epochs}
def predict_samples_in_batches(model_to_predict_batches, samples_to_predict_in_batches, batch_size=20):
    scores, ranks, predictions = [], [], []

    start = 0
    while start < len(samples_to_predict_in_batches):
        end = min(len(samples_to_predict_in_batches), start + batch_size)

        cur_batch = numpy.array(samples_to_predict_in_batches[start:end])
        cur_batch_scores, cur_batch_ranks, cur_batch_predictions = model_to_predict_batches.predict_samples(cur_batch)
        scores += cur_batch_scores
        ranks += cur_batch_ranks
        predictions += cur_batch_predictions

        start += batch_size

    return scores, ranks, predictions

original_model = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)  # type: TransE
original_model.to('cuda')
original_model.load_state_dict(torch.load(args.model_path))
original_model.eval()
with open(args.result_path,'r') as f:
    input_lines = f.readlines()
i = 0
facts_to_explain = []
samples_to_explain = []
perspective = "head"    # for all samples the perspective was head for simplicity
sample_to_explain_2_best_rule = {}
final_path = {}
index = 0
while i <= len(input_lines) -3:
    fact_line = input_lines[i]
    rules_line = input_lines[i + 1]
    empty_line = input_lines[i + 2]
    rules_str = rules_line.split(':')[:-1]
    final_triplets = set()
    count = 0

    a = 0
    tp_num = set()

    for rule in rules_str:
        triplets = re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", rule.split('[')[1])
        exp = []
        for tp in triplets:
            ent1, rel1, ent2 = tp
            n_ent1 = dataset.get_name_for_entity_id(int(ent1))
            n_rel1 = dataset.get_name_for_relation_id(int(rel1))
            n_ent2 = dataset.get_name_for_entity_id(int(ent2))
            exp.append(f'{n_ent1}\t{n_rel1}\t{n_ent2}')
            # if tp in sel_head_list:
            final_triplets.add(tp)
        a += 1
            # break
        fact = tuple(fact_line.strip().split(";"))
        if f'{fact[0]}\t{fact[1]}\t{fact[2]}' not in final_path:
            final_path[f'{fact[0]}\t{fact[1]}\t{fact[2]}'] = [exp]
        else:
            final_path[f'{fact[0]}\t{fact[1]}\t{fact[2]}'].append(exp)
        count += len(triplets)
        if a >= 1:
            break

    if rules_line.strip() != '':
        fact = tuple(fact_line.strip().split(";"))
        facts_to_explain.append(fact)
        sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
        samples_to_explain.append(sample)
    # 将结果转换为整数的三元组列表
    #     rule_samples = [tuple(map(int, final_triplets)) ]
        rule_samples = [tuple(map(int, triplet)) for triplet in final_triplets]
        sample_to_explain_2_best_rule[sample] = rule_samples
    i += 3
    index += 1
samples_to_remove = []  # the samples to remove from the training set before retraining
for sample_to_explain in samples_to_explain:
    best_rule_samples = sample_to_explain_2_best_rule[sample_to_explain]
    samples_to_remove += best_rule_samples
samples_to_remove = list(set(samples_to_remove))
print(len(samples_to_remove))

new_dataset = copy.deepcopy(dataset)

print("Removing samples: ")
for (head, relation, tail) in samples_to_remove:
    print("\t" + dataset.printable_sample((head, relation, tail)))

# remove the samples_to_remove from training samples of new_dataset (and update new_dataset.to_filter accordingly)
new_dataset.remove_training_samples(numpy.array(samples_to_remove))

# obtain tail ranks and scores of the original model for all samples_to_explain
# original_scores, original_ranks, original_predictions = original_model.predict_samples(numpy.array(samples_to_explain))
original_scores, original_ranks, original_predictions = predict_samples_in_batches(original_model, samples_to_explain)

######

new_model = TransE(dataset=new_dataset,
                   hyperparameters=hyperparameters,
                   init_random=True)  # type: TransE
new_optimizer = PairwiseRankingOptimizer(model=new_model, hyperparameters=hyperparameters)
new_optimizer.train(train_samples=new_dataset.train_samples)
new_model.eval()

# new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))
new_scores, new_ranks, new_predictions = predict_samples_in_batches(new_model, samples_to_explain)

for i in range(len(samples_to_explain)):
    cur_sample = samples_to_explain[i]
    original_direct_score = original_scores[i][0]
    print(1)
    print(original_direct_score)
    original_tail_rank = original_ranks[i][1]

    new_direct_score = new_scores[i][0]
    print(new_direct_score)
    new_tail_rank = new_ranks[i][1]

    print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                           dataset.relation_id_2_name[cur_sample[1]],
                           dataset.entity_id_2_name[cur_sample[2]]]) + ">")
    print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_direct_score))
    print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank))
    print()

output_lines = []
for i in range(len(samples_to_explain)):
    cur_sample_to_explain = samples_to_explain[i]

    original_direct_score = original_scores[i][0]
    original_tail_rank = original_ranks[i][1]

    new_direct_score = new_scores[i][0]
    new_tail_rank = new_ranks[i][1]

    # original_head, original_relation, original_tail,
    # fact_to_convert_head, fact_to_convert_rel, fact_to_convert_tail,

    # original_direct_score, new_direct_score,
    #  original_tail_rank, new_tail_rank

    a = ";".join(dataset.sample_to_fact(cur_sample_to_explain))

    b = []
    samples_to_remove_from_this_entity = sample_to_explain_2_best_rule[cur_sample_to_explain]
    for x in range(4):
        if x < len(samples_to_remove_from_this_entity):
            b.append(";".join(dataset.sample_to_fact(samples_to_remove_from_this_entity[x])))
        else:
            b.append(";;")

    b = ";".join(b)
    c = str(original_direct_score) + ";" + str(new_direct_score)
    d = str(original_tail_rank) + ";" + str(new_tail_rank)
    output_lines.append(";".join([a, b, c, d]) + "\n")
with open("pathX_transe_fb15k237.csv", "w") as outfile:
    outfile.writelines(output_lines)

    print()