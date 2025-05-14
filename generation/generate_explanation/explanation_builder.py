import itertools
import json
import random
from typing import Tuple, Any
from collections import defaultdict
from dataset import Dataset
from local_train import LPFT_Engine
from link_prediction.models.model import Model
import numpy
from prefilters.no_prefilter import NoPreFilter
from prefilters.prefilter import TYPE_PREFILTER, TOPOLOGY_PREFILTER, NO_PREFILTER
from prefilters.type_based_prefilter import TypeBasedPreFilter
from prefilters.topology_prefilter import TopologyPreFilter

DEAFAULT_XSI_THRESHOLD = 5

class ExplanationBuilder:
    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 sample_to_explain: Tuple[Any, Any, Any]):
        self.model = model
        self.dataset = dataset
        self.sample_to_explain = sample_to_explain
        self.triple_to_explain = self.dataset.sample_to_fact(self.sample_to_explain)

class ModelExplanationBuilder(ExplanationBuilder):

    def __init__(self, model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 sample_to_explain: Tuple[Any, Any, Any]
                 ):

        super().__init__(model=model, dataset=dataset,
                         sample_to_explain=sample_to_explain)
        self.window_size = 10
        self.engine = LPFT_Engine(model=model,
                                  dataset=dataset,
                                  hyperparameters=hyperparameters)

    def build_explanations(self,
                           samples_to_remove: list,
                           top_k: int = 10):

        all_rules_with_relevance = []
        sample_2_relevance_len1 = {}
        sample_2_relevance_len2 = {}
        sample_2_relevance_len3 = {}
        sample_2_relevance = {}

        head_count = defaultdict(int)
        for sample_to_remove in samples_to_remove:
            if len(sample_to_remove) > 1:
                head_count[sample_to_remove[0]] += 1

        # this is an exception: all samples (= rules with length 1) are tested
        for i, sample_to_remove in enumerate(samples_to_remove):
            relevance, score, score_ori, emb1, emb2 = self._compute_relevance_for_path(sample_to_remove, head_count)
            sample_2_relevance[str(sample_to_remove)] = relevance
            if len(sample_to_remove) == 1:
                sample_2_relevance_len1[str(sample_to_remove)] = relevance
            if len(sample_to_remove) == 2:
                sample_2_relevance_len2[str(sample_to_remove)] = relevance
            if len(sample_to_remove) == 3:
                sample_2_relevance_len3[str(sample_to_remove)] = relevance
            print("\tObtained relevance: " + str(relevance))
        samples_with_relevance = sorted(sample_2_relevance.items(), key=lambda x: x[1], reverse=True)
        return sorted(samples_with_relevance, key=lambda x: x[1], reverse=True), score, score_ori, emb1, emb2

    def _compute_relevance_for_path(self, nple_to_remove: list, head_count):

        assert (len(nple_to_remove[0]) == 3)

        relevance, \
        original_best_entity_score, original_target_entity_score, original_target_entity_rank, \
        base_pt_best_entity_score, base_pt_target_entity_score, base_pt_target_entity_rank, \
        pt_best_entity_score, pt_target_entity_score, pt_target_entity_rank, execution_time, emb1, emb2 = \
            self.engine.removal_relevance(sample_to_explain=self.sample_to_explain,
                                          samples_to_remove=nple_to_remove, head_count=head_count)

        return relevance, base_pt_target_entity_score, pt_target_entity_score,emb1, emb2

    def _preliminary_rule_score(self, rule, sample_2_relevance):
        return numpy.sum([sample_2_relevance[x] for x in rule])

    def _average(self, l: list):
        result = 0.0
        for item in l:
            result += float(item)
        return result / float(len(l))

class PathX:
    """
    The PathX object is the overall manager of the explanation process.
    """

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict,
                 prefilter_type: str,
                 relevance_threshold: float = None
                ):
        """
        PathX object constructor.
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.relevance_threshold = relevance_threshold
        if prefilter_type == TOPOLOGY_PREFILTER:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)
        elif prefilter_type == TYPE_PREFILTER:
            self.prefilter = TypeBasedPreFilter(model=model, dataset=dataset)
        elif prefilter_type == NO_PREFILTER:
            self.prefilter = NoPreFilter(model=model, dataset=dataset)
        else:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)

        self.engine = LPFT_Engine(model=model,
                                  dataset=dataset,
                                  hyperparameters=hyperparameters)

    def explain_paths(self,
                      sample_to_explain: Tuple[Any, Any, Any], path
                      ):

        for target, paths in path.items():
            if target == sample_to_explain:
                most_promising_samples = paths

        explanation_builder = ModelExplanationBuilder(model=self.model,
                                                      dataset=self.dataset,
                                                      hyperparameters=self.hyperparameters,
                                                      sample_to_explain=sample_to_explain
                                                      )

        explanations_with_relevance, score, score_new,emb1, emb2 = explanation_builder.build_explanations(samples_to_remove=most_promising_samples,top_k=50)
        return explanations_with_relevance
