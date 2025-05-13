import copy
from collections import defaultdict
import numpy
from dataset import Dataset
import json
# from get_subgraph import sel_influential_rels
# from get_subgraph import extract_influential_subgraph
# from get_subgraph import find_paths
from get_subgraph_2 import sel_influential_rels
from get_subgraph_2 import extract_influential_subgraph
from get_subgraph_2 import find_paths
class pathXDataset(Dataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        the pathXDataset has the responsibility to decide the id of the pathX entity (aka mimic in our paper)
        and to store the train, valid and test samples specific for the original entity and for the pathX entity

        A pathXDataset is never *loaded* from file: it is always generated from a pre-existing, already loaded Dataset.

        Nomenclature used in the pathXDataset:
            - "original entity": the entity to explain the prediction of in the original Dataset;
            - "clone entity": a homologous mimic, i.e., a "fake" entity
                              post-trained with the same training samples as the original entity
            - "pathX entity": a non-homologous mimic, i.e., a "fake" entity
                               post-trained with slightly different training samples from the original entity.
                               (e.g. some training samples may have been removed, or added).
    """

    def __init__(self,
                 dataset: Dataset,
                 entity_id: int,
                 entity_id2: int,
                 rel_id: int
               ):

        super(pathXDataset, self).__init__(name=dataset.name,
                                            separator=dataset.separator,
                                            load=False)

        if dataset.num_entities == -1:
            raise Exception("The Dataset passed to initialize a pathXDataset must be already loaded")

        # the pathXDataset is now basically empty (because load=False was used in the super constructor)
        # so we must manually copy (and sometimes update) all the important attributes from the original loaded Dataset
        self.num_entities = dataset.num_entities + 1                # adding the pathX entity to the count
        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations

        # copy relevant data structures
        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.train_to_filter = copy.deepcopy(dataset.train_to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)

        # add the pathX entity
        self.original_entity_id = entity_id
        # self.original_entity_id2 = entity_id2
        # self.original_rel_id = rel_id

        self.original_entity_name = self.entity_id_2_name[self.original_entity_id]
        # self.original_entity_name2 = self.entity_id_2_name[self.original_entity_id2]
        # self.original_rel_name = self.relation_id_2_name[self.original_rel_id]

        self.pathX_entity_id = dataset.num_entities  # add the pathX entity to the dataset; it is always the last one
        # self.pathX_entity_id2 = dataset.num_entities + 1
        # self.pathX_rel_id = dataset.num_relations
        self.pathX_entity_name = "pathX_" + self.original_entity_name
        # self.pathX_entity_name2 = "pathX_" + self.original_entity_name2
        # self.pathX_rel_name = "pathX_" + self.original_rel_name
        self.entity_name_2_id[self.pathX_entity_name] = self.pathX_entity_id
        # self.entity_name_2_id[self.pathX_entity_name2] = self.pathX_entity_id2
        # self.relation_name_2_id[self.pathX_rel_name] = self.pathX_rel_id
        self.entity_id_2_name[self.pathX_entity_id] = self.pathX_entity_name
        # self.entity_id_2_name[self.pathX_entity_id2] = self.pathX_entity_name2
        # self.relation_id_2_name[self.pathX_rel_id] = self.pathX_rel_name

        # We do not copy all the triples and samples from the original dataset: the pathXDataset DOES NOT NEED THEM.
        # The train, valid, and test samples of the pathXDataset are generated using only those that featured the original entity!
        self.original_train_samples = self.get_subgraph(dataset.train_samples, entity_id, entity_id2, rel_id)
        self.original_valid_samples = dataset.valid_samples
        self.original_test_samples =dataset.test_samples
        # self.original_train_samples = self._extract_samples_with_entity(dataset.train_samples, self.original_entity_id)
        # self.original_valid_samples = self._extract_samples_with_entity(dataset.valid_samples, self.original_entity_id)
        # self.original_test_samples = self._extract_samples_with_entity(dataset.test_samples, self.original_entity_id)
        # self.original_valid_samples = self.get_subgraph(dataset.valid_samples, entity_id, entity_id2, rel_id)
        # self.original_test_samples = self.get_subgraph(dataset.test_samples, entity_id, entity_id2, rel_id)
        self.pathX_train_samples = Dataset.replace_entity_rel_in_samples(self.original_train_samples, self.original_entity_id, self.pathX_entity_id)

        self.pathX_valid_samples = Dataset.replace_entity_rel_in_samples(self.original_valid_samples, self.original_entity_id, self.pathX_entity_id)

        self.pathX_test_samples = Dataset.replace_entity_rel_in_samples(self.original_test_samples, self.original_entity_id, self.pathX_entity_id)

        # update to_filter and train_to_filter data structures
        samples_to_stack = [self.pathX_train_samples]
        if len(self.pathX_valid_samples) > 0:
            samples_to_stack.append(self.pathX_valid_samples)
        if len(self.pathX_test_samples) > 0:
            samples_to_stack.append(self.pathX_test_samples)
        all_pathX_samples = numpy.vstack(samples_to_stack)
        for i in range(all_pathX_samples.shape[0]):
            (head_id, relation_id, tail_id) = all_pathX_samples[i]
            self.to_filter[(head_id, relation_id)].append(tail_id)
            self.to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)
            # if the sample was a training sample, also do the same for the train_to_filter data structure;
            # Also fill the entity_2_degree and relation_2_degree dicts.
            if i < len(self.pathX_train_samples):
                self.train_to_filter[(head_id, relation_id)].append(tail_id)
                self.train_to_filter[(tail_id, relation_id + self.num_direct_relations)].append(head_id)

        # create a map that associates each pathX train_sample to its index in self.pathX_train_samples
        # this will be necessary to allow efficient removals and undoing removals
        self.pathX_train_sample_2_index = {}
        for i in range(len(self.pathX_train_samples)):
            cur_head, cur_rel, cur_tail = self.pathX_train_samples[i]
            self.pathX_train_sample_2_index[(cur_head, cur_rel, cur_tail)] = i


        # initialize data structures needed in the case of additions and/or removals;
        # these structures are required to undo additions and/or removals
        self.pathX_train_samples_copy = copy.deepcopy(self.pathX_train_samples)
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_pathX_samples = []

        self.last_removed_samples = []
        self.last_removed_samples_number = 0
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_pathX_samples = []

    # override
    def remove_training_samples(self, samples_to_remove: numpy.array):
        """
            Remove some training samples from the pathX training samples of this pathXDataset.
            The samples to remove must still feature the original entity id; this method will convert them before removal.
            The pathXDataset will keep track of the last performed removal so it can be undone if necessary.

            :param samples_to_remove: the samples to add, still featuring the id of the original entity,
                                   in the form of a numpy array
        """


        self.last_removed_samples = samples_to_remove
        self.last_removed_samples_number = len(samples_to_remove)

        # reset data structures needed to undo removals. We only want to keep track of the *last* removal.
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_pathX_samples = []

        pathX_train_samples_to_remove = Dataset.replace_entity_rel_in_samples(samples=samples_to_remove,
                                                                               old_entity1=self.original_entity_id,
                                                                               new_entity1=self.pathX_entity_id,
                                                                               as_numpy=False)

        # update to_filter and train_to_filter
        for (cur_head, cur_rel, cur_tail) in pathX_train_samples_to_remove:
            if cur_tail in self.to_filter[(cur_head, cur_rel)]:
                self.to_filter[(cur_head, cur_rel)].remove(cur_tail)
                self.to_filter[(cur_tail, cur_rel + self.num_direct_relations)].remove(cur_head)
                self.train_to_filter[(cur_head, cur_rel)].remove(cur_tail)
                self.train_to_filter[(cur_tail, cur_rel + self.num_direct_relations)].remove(cur_head)

                # and also update the data structures required for undoing the removal
                self.last_removed_pathX_samples.append((cur_head, cur_rel, cur_tail))
                self.last_filter_removals[(cur_head, cur_rel)].append(cur_tail)
                self.last_filter_removals[(cur_tail, cur_rel + self.num_direct_relations)].append(cur_head)

        # get the indices of the samples to remove in the pathX_train_samples structure
        # and use them to perform the actual removal
        pathX_train_indices_to_remove = [self.pathX_train_sample_2_index[x] for x in pathX_train_samples_to_remove]
        self.pathX_train_samples = numpy.delete(self.pathX_train_samples, pathX_train_indices_to_remove, axis=0)


    def undo_last_training_samples_removal(self):
        """
            This method undoes the last removal performed on this pathXDataset
            calling its add_training_samples method.

            The purpose of undoing the removals performed on a pre-existing pathXDataset,
            instead of creating a new pathXDataset from scratch, is to improve efficiency.
        """
        if self.last_removed_samples_number <= 0:
            raise Exception("No removal to undo.")

        # revert the self.pathX_train_samples to the self.pathX_train_samples_copy
        self.pathX_train_samples = copy.deepcopy(self.pathX_train_samples_copy)

        # undo additions to to_filter and train_to_filter
        for key in self.last_filter_removals:
            for x in self.last_filter_removals[key]:
                self.to_filter[key].append(x)
                self.train_to_filter[key].append(x)

        # reset the data structures used to undo additions
        self.last_removed_samples = []
        self.last_removed_samples_number = 0
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_pathX_samples = []


    def as_pathX_sample(self, original_sample):
        if not self.original_entity_id in original_sample:
            raise Exception("Could not find the original entity " + str(self.original_entity_id) + " in the passed sample " + str(original_sample))
        return Dataset.replace_entity_in_sample(sample=original_sample,
                                                old_entity1=self.original_entity_id,
                                                new_entity1=self.pathX_entity_id)


    def get_subgraph(self,train_samples, ent1, ent2, rel):
        csv_file = "WN18RR.csv"
        json_file = "WN18RR_relation_to_number.json"
        # csv_file = "FB15k-237.csv"  # 输入的 CSV 文件路径
        # json_file = "FB15k-237_relation_to_number.json"  # 输入的 JSON 文件路径
        influ_rels = sel_influential_rels(csv_file, json_file, rel, 1)
        get_subgraph = extract_influential_subgraph(train_samples, ent1, ent2, influ_rels)
        one_hop = find_paths(get_subgraph,ent1,ent2)
        head = self.entity_id_2_name[ent1]
        relation = self.relation_id_2_name[rel]
        tail = self.entity_id_2_name[ent2]
        with open("/data/zhaotianzhe/ryc/KGEvaluator/generation/WN18RR/WN18RR_candi_path_1_complex.json",'r') as f:
            path = json.load(f)
        candi_path = path[f'{head}\t{relation}\t{tail}']
        final_triple = set()
        for p in candi_path:
            for tp in p:
                parts = tp.strip().split('\t')
                h = self.entity_name_2_id[parts[0]]
                r = self.relation_name_2_id[parts[1]]
                t = self.entity_name_2_id[parts[2]]
                final_triple.add((h, r, t))
        result = final_triple.union(one_hop)
        # return path
        # return final_triple
        return result

        ### private utility methods
    @staticmethod
    def _extract_samples_with_entity(samples, entity_id):
        return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]