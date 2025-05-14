import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import random
import time
import json
import numpy
import torch

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

from dataset import ALL_DATASET_NAMES, Dataset
from explanation_builder import PathX as PathX
from link_prediction.models.complex import ComplEx
from link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_1, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, \
    BATCH_SIZE, REGULARIZER_NAME

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
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

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

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
torch.backends.cudnn.deterministic = True
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())

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

prefilter = args.prefilter

# load the dataset and its training samples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(name=args.dataset, separator="\t", load=True)

print("Reading facts to explain...")
with open(args.facts_to_explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]

model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)  # type: ComplEx
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
    key_id = (dataset.get_id_for_entity_name(key.split("\t")[0]),dataset.get_id_for_relation_name(key.split("\t")[1]),dataset.get_id_for_entity_name(key.split("\t")[2]))

    value_ids = []
    for path in value:
        path_id = []
        for i, tp in enumerate(path):
            tp_ids = (dataset.get_id_for_entity_name(path[i].split("\t")[0]),dataset.get_id_for_relation_name(path[i].split("\t")[1]),dataset.get_id_for_entity_name(path[i].split("\t")[2]))
            path_id.append(tp_ids)
        value_ids.append(path_id)
    candidate_path_id[key_id] = value_ids
if args.baseline is None:
    pathX = PathX(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter)
for i, fact in enumerate(testing_facts):
    if i>= 0:
        head, relation, tail = fact
        print("Explaining fact " + str(i) + " on " + str(
            len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
        head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
                                        dataset.get_id_for_relation_name(relation), \
                                        dataset.get_id_for_entity_name(tail)
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
with open("pathX_complex_wn.txt".format(args.baseline), "w") as output:

    output.writelines(output_lines)
