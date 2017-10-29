import json
import os
import random
import itertools
import math
from collections import defaultdict

import numpy as np

from my.tensorflow import grouper
from my.utils import index
from basic.create_shared import create_shared

class Data(object):
    def get_size(self):
        raise NotImplementedError()

    def get_by_idxs(self, idxs):
        """
        Efficient way to obtain a batch of items from filesystem
        :param idxs:
        :return dict: {'X': [,], 'Y', }
        """
        data = defaultdict(list)
        for idx in idxs:
            each_data = self.get_one(idx)
            for key, val in each_data.items():
                data[key].append(val)
        return data

    def get_one(self, idx):
        raise NotImplementedError()

    def get_empty(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()


class DataSet(object):
    def __init__(self, data, data_type, shared_path, shared = None, load_shared = True, valid_idxs=None):
        self.data = data  # e.g. {'X': [0, 1, 2], 'Y': [2, 3, 4]}
        self.data_type = data_type
        self.shared_path = shared_path
        self.shared = shared
        total_num_examples = self.get_data_size()
        self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
        self.num_examples = len(self.valid_idxs)
        self.load_shared = load_shared

    def _sort_key(self, idx):
        rx = self.data['*x'][idx]
        x = self.shared['x'][rx[0]][rx[1]]
        return max(map(len, x))

    def get_data_size(self):
        if isinstance(self.data, dict):
            return len(next(iter(self.data.values())))
        elif isinstance(self.data, Data):
            return self.data.get_size()
        raise Exception()

    def get_by_idxs(self, idxs):
        if isinstance(self.data, dict):
            out = defaultdict(list)
            for key, val in self.data.items():
                out[key].extend(val[idx] for idx in idxs)
            return out
        elif isinstance(self.data, Data):
            return self.data.get_by_idxs(idxs)
        raise Exception()

    def get_batches(self, batch_size, num_batches=None, shuffle=False, cluster=False):
        """

        :param batch_size:
        :param num_batches:
        :param shuffle:
        :param cluster: cluster examples by their lengths; this might give performance boost (i.e. faster training).
        :return:
        """
        num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
        if num_batches is None:
            num_batches = num_batches_per_epoch
        num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

        if shuffle:
            random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
            if cluster:
                sorted_idxs = sorted(random_idxs, key=self._sort_key)
                sorted_grouped = lambda: list(grouper(sorted_idxs, batch_size))
                grouped = lambda: random.sample(sorted_grouped(), num_batches_per_epoch)
            else:
                random_grouped = lambda: list(grouper(random_idxs, batch_size))
                grouped = random_grouped
        else:
            raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
            grouped = raw_grouped

        batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))

        def get_shared(pos):
            return json.load(open(self.shared_path \
                    % (str(pos[0]).zfill(3), str(pos[1]).zfill(3)+".json"), 'r'))
        for _ in range(num_batches):
            batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
            batch_data = self.get_by_idxs(batch_idxs)
            shared_batch_data = {}
            
            if self.load_shared:
                pos = batch_data['*x']
                shared_list = [get_shared(each) for each in pos]
                for k in ['p', 'x', 'cx']:
                    shared_batch_data[k] = [sh[k] for sh in shared_list]
            else:
                for key, val in batch_data.items():
                    if key.startswith('*'):
                        assert self.shared is not None
                        shared_key = key[1:]
                        shared_batch_data[shared_key] = [index(self.shared[shared_key], each) for each in val]
            batch_data.update(shared_batch_data)

            batch_ds = DataSet(batch_data, self.data_type, self.shared_path, load_shared=self.load_shared, shared=self.shared)
            yield batch_idxs, batch_ds

    def get_multi_batches(self, batch_size, num_batches_per_step, num_steps=None, shuffle=False, cluster=False):
        batch_size_per_step = batch_size * num_batches_per_step
        batches = self.get_batches(batch_size_per_step, num_batches=num_steps, shuffle=shuffle, cluster=cluster)
        multi_batches = (tuple(zip(grouper(idxs, batch_size, shorten=True, num_groups=num_batches_per_step),
                         data_set.divide(num_batches_per_step))) for idxs, data_set in batches)
        return multi_batches

    def get_empty(self):
        if isinstance(self.data, dict):
            data = {key: [] for key in self.data}
        elif isinstance(self.data, Data):
            data = self.data.get_empty()
        else:
            raise Exception()
        return DataSet(data, self.data_type, self.shared_path, load_shared=self.load_shared, shared=self.shared)

    def __add__(self, other):
        if isinstance(self.data, dict):
            data = {key: val + other.data[key] for key, val in self.data.items()}
        elif isinstance(self.data, Data):
            data = self.data + other.data
        else:
            raise Exception()

        valid_idxs = list(self.valid_idxs) + [valid_idx + self.num_examples for valid_idx in other.valid_idxs]
        return DataSet(data, self.data_type, self.shared_path, shared=self.shared, valid_idxs=valid_idxs, load_shared=self.load_shared)

    def divide(self, integer):
        batch_size = int(math.ceil(self.num_examples / integer))
        idxs_gen = grouper(self.valid_idxs, batch_size, shorten=True, num_groups=integer)
        data_gen = (self.get_by_idxs(idxs) for idxs in idxs_gen)
        ds_tuple = tuple(DataSet(data, self.data_type, self.shared_path, shared=self.shared, load_shared=self.load_shared) for data in data_gen)
        return ds_tuple


def load_metadata(config, data_type):
    metadata_path = os.path.join(config.data_dir, "metadata_{}.json".format(data_type))
    with open(metadata_path, 'r') as fh:
        metadata = json.load(fh)
        for key, val in metadata.items():
            config.__setattr__(key, val)
        return metadata


def read_data(config, data_type, ref, data_filter=None):

    data_path = os.path.join(config.data_dir, "data_{}.json".format(data_type))
    _shared_path = config.data_dir + "/shared_" + data_type + "_%s_%s" 
    with open(data_path, 'r') as fh:
        data = json.load(fh)
    #if config.load_shared:
    #    shared = {}
    #else:
    shared_path = os.path.join(config.data_dir, "shared_{}.json".format(data_type))
    if not config.using_shared:
        with open(shared_path, 'r') as fh:
            shared = json.load(fh)
    else:
        shared = {}
    num_examples = len(next(iter(data.values())))
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))

    shared_path = config.shared_path or os.path.join(config.out_dir, "shared.json")
    if not (ref or config.using_shared): create_shared(config, shared, shared_path)

    new_shared = json.load(open(shared_path, 'r'))

    for (k, v) in new_shared.items():
        shared[k] = v
    if config.use_glove_for_unk:
        # create new word2idx and word2vec
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
        shared['new_word2idx'] = new_word2idx_dict
        offset = len(shared['word2idx'])
        word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
        new_word2idx_dict = shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        shared['new_emb_mat'] = new_emb_mat

    data_set = DataSet(data, data_type, _shared_path, shared=shared, load_shared=config.load_shared, valid_idxs=valid_idxs)
    return data_set


def get_squad_data_filter(config):
    def data_filter(data_point):
        rx, rcx, q, cq, y = (data_point[key] for key in ('*x', '*cx', 'q', 'cq', 'y'))
        if len(q) > config.ques_size_th:
            return False
        is_span = config.model_name in ['basic', 'span-gen']
        is_gen = config.model_name in ['basic-generate', 'span-gen', 'baseline']
        is_marco = config.data_dir.startswith('data/msmarco')

        if is_gen: return True # for SQUAD
        if is_gen and (not is_marco):
            y_ = y if not config.model_name == 'span-gen' else y[1]
            if (config.model_name is not 'baseline') and 1 in y_: return False
        if config.model_name == 'baseline' and is_marco and y[1][1]==[]: return False
        if not is_span: return True
        if is_marco: y = y[1][0]
        if config.model_name.startswith('data/newsqa/class'): y =y[1]
        if config.model_name == 'basic' and y == []: return False

        if config.squash:
            for start, stop in y:
                stop_offset = sum(map(len, xi[:stop[0]]))
                if stop_offset + stop[1] > config.para_size_th:
                    return False
            return True

        if config.single:
            for start, stop in y:
                if start[0] != stop[0]:
                    return False
        if config.data_filter == 'max':
            for start, stop in y:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != stop[0]:
                    return False
                if stop[1] >= config.sent_size_th:
                    return False
        elif config.data_filter == 'valid':
            if len(xi) > config.num_sents_th:
                return False
            if any(len(xij) > config.sent_size_th for xij in xi):
                return False
        elif config.data_filter == 'semi':
            """
            Only answer sentence needs to be valid.
            """
            for start, stop in y:
                if stop[0] >= config.num_sents_th:
                    return False
                if start[0] != start[0]:
                    return False
                if len(xi[start[0]]) > config.sent_size_th:
                    return False
        else:
            raise Exception()

        return True
    return data_filter


def update_config(config, data_sets):
    config.max_num_sents = config.num_sents_th
    config.max_sent_size = config.sent_size_th
    config.max_ques_size = config.ques_size_th
    config.max_word_size = config.word_size_th
    config.max_para_size = config.para_size_th

    """
    for data_set in data_sets:
        data = data_set.data
        shared = data_set.shared
        for idx in data_set.valid_idxs:
            rx = data['*x'][idx]
            q = data['q'][idx]
            sents = shared['x'][rx[0]][rx[1]]
            config.max_para_size = max(config.max_para_size, sum(map(len, sents)))
            config.max_num_sents = max(config.max_num_sents, len(sents))
            config.max_sent_size = max(config.max_sent_size, max(map(len, sents)))
            config.max_word_size = max(config.max_word_size, max(len(word) for sent in sents for word in sent))
            if len(q) > 0:
                config.max_ques_size = max(config.max_ques_size, len(q))
                config.max_word_size = max(config.max_word_size, max(len(word) for word in q))

    if config.mode == 'train':
        config.max_num_sents = min(config.max_num_sents, config.num_sents_th)
        config.max_sent_size = min(config.max_sent_size, config.sent_size_th)
        config.max_para_size = min(config.max_para_size, config.para_size_th)

    config.max_word_size = min(config.max_word_size, config.word_size_th)
    """

    config.char_vocab_size = len(data_sets[0].shared['char2idx'])
    config.word_emb_size = len(next(iter(data_sets[0].shared['lower_word2vec'].values())))
    config.word_vocab_size = len(data_sets[0].shared['word2idx'])

    if config.single:
        config.max_num_sents = 1
    if config.squash:
        config.max_sent_size = config.max_para_size
        config.max_num_sents = 1
