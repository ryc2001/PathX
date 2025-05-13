import sys
import argparse
import random
import time
import json
import numpy
import torch
import os

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

# Import dataset and model definitions
from dataset import Dataset, ALL_DATASET_NAMES
from explanation_builder import PathX as PathX
from link_prediction.models.transe import TransE
from link_prediction.models.model import BATCH_SIZE, LEARNING_RATE, EPOCHS, DIMENSION, MARGIN, NEGATIVE_SAMPLES_RATIO, \
    REGULARIZER_WEIGHT
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER

# Argument parser setup
parser = argparse.ArgumentParser()

# Dataset selection
parser.add_argument("--dataset",
                    type=str,
                    choices=ALL_DATASET_NAMES,
                    help="The dataset to use: FB15k-237 or WN18RR")

# Model and training parameters
parser.add_argument("--max_epochs", type=int, default=1000, help="Number of epochs.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate.")
parser.add_argument("--dimension", type=int, default=200, help="Embedding dimensionality.")
parser.add_argument("--margin", type=int, default=5, help="Margin for pairwise ranking loss.")
parser.add_argument("--negative_samples_ratio", type=int, default=3, help="Negative sample ratio.")
parser.add_argument("--regularizer_weight", type=float, default=0.0, help="L2 regularization weight.")

# File paths
parser.add_argument("--model_path", type=str, help="Path to the trained model to explain.")
parser.add_argument("--facts_to_explain_path", type=str, help="Path to the test facts to be explained.")
parser.add_argument("--entities_to_convert", type=str, help="Entity file for baseline explanation (optional).")
parser.add_argument("--candidate_path_dict", type=str, help="Path to candidate paths for target facts.")

args = parser.parse_args()

# Set random seeds for reproducibility
seed = 42
torch.backends.cudnn.deterministic = True
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())

# Pack hyperparameters for model loading
hyperparameters = {
    DIMENSION: args.dimension,
    MARGIN: args.margin,
    NEGATIVE_SAMPLES_RATIO: args.negative_samples_ratio,
    REGULARIZER_WEIGHT: args.regularizer_weight,
    BATCH_SIZE: args.batch_size,
    LEARNING_RATE: args.learning_rate,
    EPOCHS: args.max_epochs
}

relevance_threshold = args.relevance_threshold
prefilter = args.prefilter

# ========== Load dataset ==========
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

# ========== Load facts to explain ==========
print("Reading facts to explain...")
with open(args.facts_to_explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

# ========== Load model ==========
model = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
model.to('cuda')
model.load_state_dict(torch.load(args.model_path))
model.eval()

# ========== Initialize explanation module ==========
if args.baseline is None:
    PathX = PathX(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        prefilter_type=prefilter,
        relevance_threshold=relevance_threshold
    )

# ========== Load candidate paths ==========
with open(args.candidate_path_dict, 'r') as f:
    candidate_path_dict = json.load(f)

candidate_path_id = {}

# Convert string-based path triples to ID-based triples
for key, value in candidate_path_dict.items():
    key_id = (
        dataset.get_id_for_entity_name(key.split("\t")[0]),
        dataset.get_id_for_relation_name(key.split("\t")[1]),
        dataset.get_id_for_entity_name(key.split("\t")[2])
    )
    value_ids = []
    for path in value:
        path_id = []
        for i, tp in enumerate(path):
            tp_ids = (
                dataset.get_id_for_entity_name(path[i].split("\t")[0]),
                dataset.get_id_for_relation_name(path[i].split("\t")[1]),
                dataset.get_id_for_entity_name(path[i].split("\t")[2])
            )
            path_id.append(tp_ids)
        value_ids.append(path_id)
    candidate_path_id[key_id] = value_ids

# ========== Run PathX explanation for each test fact ==========
output_lines = []
start_time = time.time()

for i, fact in enumerate(testing_facts):
    head, relation, tail = fact
    print(f"Explaining fact {i} of {len(testing_facts)}: <{head}, {relation}, {tail}>")

    # Convert fact to ID form
    head_id = dataset.get_id_for_entity_name(head)
    relation_id = dataset.get_id_for_relation_name(relation)
    tail_id = dataset.get_id_for_entity_name(tail)
    sample_to_explain = (head_id, relation_id, tail_id)

    # Run explanation
    rule_samples_with_relevance = PathX.explain_paths(
        sample_to_explain=sample_to_explain,
        path=candidate_path_id
    )

    # Format explanation outputs
    rule_facts_with_relevance = []
    for cur_rule_with_relevance in rule_samples_with_relevance:
        cur_rule_samples, cur_relevance = cur_rule_with_relevance
        rule_facts_with_relevance.append(cur_rule_samples + ":" + str(cur_relevance))

    output_lines.append(";".join(fact) + "\n")
    output_lines.append(",".join(rule_facts_with_relevance) + "\n")
    output_lines.append("\n")

end_time = time.time()
print("Required time: %.2f seconds" % (end_time - start_time))

# ========== Write explanation results to file ==========
with open("pathx_transe_fb.txt", "w") as output:
    output.writelines(output_lines)
