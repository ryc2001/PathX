import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

import argparse
import copy
import random
import re
import numpy
import torch
import json

from dataset import ALL_DATASET_NAMES, Dataset, MANY_TO_ONE, ONE_TO_ONE
from link_prediction.models.complex import ComplEx
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, REGULARIZER_WEIGHT, EPOCHS, \
    BATCH_SIZE, REGULARIZER_NAME

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--batch_size',
                    default=100,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--max_epochs',
                    default=200,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate")

parser.add_argument('--reg',
                    default=0,
                    type=float,
                    help="Regularization weight")

parser.add_argument('--init',
                    default=1e-3,
                    type=float,
                    help="Initial scale")

parser.add_argument('--decay1',
                    default=0.9,
                    type=float,
                    help="Decay rate for the first moment estimate in Adam")

parser.add_argument('--decay2',
                    default=0.999,
                    type=float,
                    help="Decay rate for second moment estimate in Adam")
parser.add_argument("--result_path",
                    type=str,
                    help="Path where to find the model to explain")

args = parser.parse_args()

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

#############  LOAD DATASET
# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

# get the ids of the elements of the fact to explain and the perspective entity
hyperparameters = {DIMENSION: args.dimension,
                   INIT_SCALE: args.init,
                   LEARNING_RATE: args.learning_rate,
                   OPTIMIZER_NAME: args.optimizer,
                   DECAY_1: args.decay1,
                   DECAY_2: args.decay2,
                   REGULARIZER_WEIGHT: args.reg,
                   EPOCHS: args.max_epochs,
                   BATCH_SIZE: args.batch_size,
                   REGULARIZER_NAME: "N3"}

original_model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: ComplEx
original_model.to('cuda')
original_model.load_state_dict(torch.load(args.model_path))
original_model.eval()

with open(args.result_path ,'r') as f:
    input_lines = f.readlines()
i = 0
facts_to_explain = []
samples_to_explain = []
perspective = "head"    # for all samples the perspective was head for simplicity
sample_to_explain_2_best_rule = {}
final_path = {}
while i <= len(input_lines) -3:
    fact_line = input_lines[i]
    rules_line = input_lines[i + 1]
    empty_line = input_lines[i + 2]
    rules_str = rules_line.split(':')[:-1]
    final_triplets = set()
    count = 0
    exp = []
    a = 0
    tp_num = set()
    for rule in rules_str:
        triplets = re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", rule.split('[')[1])
        a += 1
        for tp in triplets:
            ent1, rel1, ent2 = tp
            n_ent1 = dataset.get_name_for_entity_id(int(ent1))
            n_rel1 = dataset.get_name_for_relation_id(int(rel1))
            n_ent2 = dataset.get_name_for_entity_id(int(ent2))
            exp.append(f'{n_ent1}\t{n_rel1}\t{n_ent2}')
            final_triplets.add((ent1, rel1, ent2))
            # break
        count += len(triplets)
        if a >=1:
            break

    if rules_line.strip() != '':
        fact = tuple(fact_line.strip().split(";"))
        final_path[f'{fact[0]}\t{fact[1]}\t{fact[2]}'] = [exp]
        facts_to_explain.append(fact)
        sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
        samples_to_explain.append(sample)
    # 将结果转换为整数的三元组列表
        rule_samples = [tuple(map(int, triplet)) for triplet in final_triplets]
        sample_to_explain_2_best_rule[sample] = rule_samples
    i += 3
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
# remove the samples_to_remove from training samples of new_dataset (and update new_dataset.to_filter accordingly)
new_dataset.remove_training_samples(numpy.array(samples_to_remove))

# obtain tail ranks and scores of the original model for all samples_to_explain
original_scores, original_ranks, original_predictions = original_model.predict_samples(numpy.array(samples_to_explain))

######

new_model = ComplEx(dataset=new_dataset, hyperparameters=hyperparameters, init_random=True)  # type: ComplEx
new_optimizer = MultiClassNLLOptimizer(model=new_model, hyperparameters=hyperparameters)
new_optimizer.train(train_samples=new_dataset.train_samples)
new_model.eval()

new_scores, new_ranks, new_predictions = new_model.predict_samples(numpy.array(samples_to_explain))

for i in range(len(samples_to_explain)):
    cur_sample = samples_to_explain[i]
    original_direct_score = original_scores[i][0]
    original_tail_rank = original_ranks[i][1]

    new_direct_score = new_scores[i][0]
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

with open("pathX_complex_wn18rr.csv", "w") as outfile:
    outfile.writelines(output_lines)

    print()