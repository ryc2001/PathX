import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import random
import time

import numpy
import torch
import json
sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import ALL_DATASET_NAMES, Dataset

from explanation_builder import PathX as pathX
from link_prediction.models.conve import ConvE
from link_prediction.models.model import DIMENSION, LEARNING_RATE, EPOCHS, \
    BATCH_SIZE, INPUT_DROPOUT, FEATURE_MAP_DROPOUT, HIDDEN_DROPOUT, HIDDEN_LAYER_SIZE, LABEL_SMOOTHING, DECAY
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

start_time = time.time()

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--dataset",
                    type=str,
                    choices=ALL_DATASET_NAMES)

parser.add_argument("--max_epochs",
                    type=int, default=1000,
                    help="Number of epochs.")

parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size.")

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.0005,
                    help="Learning rate.")

parser.add_argument("--decay_rate",
                    type=float,
                    default=1.0,
                    help="Decay rate.")

parser.add_argument("--dimension",
                    type=int,
                    default=200,
                    help="Embedding dimensionality.")

parser.add_argument("--input_dropout",
                    type=float,
                    default=0.3,
                    nargs="?",
                    help="Input layer dropout.")

parser.add_argument("--hidden_dropout",
                    type=float,
                    default=0.4,
                    help="Dropout after the hidden layer.")

parser.add_argument("--feature_map_dropout",
                    type=float,
                    default=0.5,
                    help="Dropout after the convolutional layer.")

parser.add_argument("--label_smoothing",
                    type=float,
                    default=0.1,
                    help="Amount of label smoothing.")

parser.add_argument('--hidden_size',
                    type=int,
                    default=9728,
                    help='The side of the hidden layer. '
                         'The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

parser.add_argument("--model_path",
                    type=str,
                    required=True,
                    help="path of the model to explain the predictions of.")

parser.add_argument("--facts_to_explain_path",
                    type=str,
                    required=True,
                    help="path of the file with the facts to explain the predictions of.")

parser.add_argument("--coverage",
                    type=int,
                    default=10,
                    help="Number of random entities to extract and convert")

parser.add_argument("--entities_to_convert",
                    type=str,
                    help="path of the file with the entities to convert (only used by baselines)")
parser.add_argument("--candidate_path_dict", type=str, help="Path to candidate paths for target facts.")
args = parser.parse_args()
########## LOAD DATASET

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Reading facts to explain...")
with open(args.facts_to_explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

# get the ids of the elements of the fact to explain and the perspective entity

hyperparameters = {DIMENSION: args.dimension,
                   INPUT_DROPOUT: args.input_dropout,
                   FEATURE_MAP_DROPOUT: args.feature_map_dropout,
                   HIDDEN_DROPOUT: args.hidden_dropout,
                   HIDDEN_LAYER_SIZE: args.hidden_size,
                   BATCH_SIZE: args.batch_size,
                   LEARNING_RATE: args.learning_rate,
                   DECAY: args.decay_rate,
                   LABEL_SMOOTHING: args.label_smoothing,
                   EPOCHS: args.max_epochs}
model = ConvE(dataset=dataset,
              hyperparameters=hyperparameters,
              init_random=True)  # type: ConvE

model.to('cuda')
model.load_state_dict(torch.load(args.model_path))
model.eval()

start_time = time.time()

testing_fact_2_entities_to_convert = None

output_lines = []
with open(args.candidate_path_dict, 'r') as f:
    candidate_path_dict = json.load(f)
candidate_path_id = {}
for key, value in candidate_path_dict.items():
    # 将键转换为 ID

    key_id = (dataset.get_id_for_entity_name(key.split("\t")[0]),dataset.get_id_for_relation_name(key.split("\t")[1]),dataset.get_id_for_entity_name(key.split("\t")[2]))

    # 转换值中的每一组路径
    value_ids = []
    for path in value:
        path_id = []
        for i, tp in enumerate(path):
            tp_ids = (dataset.get_id_for_entity_name(path[i].split("\t")[0]),dataset.get_id_for_relation_name(path[i].split("\t")[1]),dataset.get_id_for_entity_name(path[i].split("\t")[2]))
            path_id.append(tp_ids)
        value_ids.append(path_id)
    candidate_path_id[key_id] = value_ids
if args.baseline is None:
    pathX = PathX(model=model, dataset=dataset, hyperparameters=hyperparameters)
for i, fact in enumerate(testing_facts):################改成target集合！命令行改
    head, relation, tail = fact
    print("Explaining fact " + str(i) + " on " + str(
        len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
    head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                    dataset.get_id_for_relation_name(relation), \
                                    dataset.get_id_for_entity_name(tail                                                              )
    sample_to_explain = (head_id, relation_id, tail_id)
    rule_samples_with_relevance = pathX.explain_paths(sample_to_explain=sample_to_explain,
                                                           path=candidate_path_id)
    rule_facts_with_relevance = []
    for cur_rule_with_relevance in rule_samples_with_relevance:
        cur_rule_samples, cur_relevance = cur_rule_with_relevance
        rule_facts_with_relevance.append(cur_rule_samples + ":" + str(cur_relevance))
    output_lines.append(";".join(fact) + "\n")
    output_lines.append(",".join(rule_facts_with_relevance) + "\n")
    output_lines.append("\n")

end_time = time.time()
print("Required time: " + str(end_time - start_time) + " seconds")
with open("pathX_conve_fb.txt".format(args.baseline), "w") as output:
    output.writelines(output_lines)
