# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-22 13:25:52
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-22 16:38:12

import logging
import multiprocessing
import os.path
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

from .bilm_interface import Bilm
from .wsi_clustering import load_embedding


class EMBED_TYPE(Enum):
    FastText = 0
    Glove = 1

embed_type = EMBED_TYPE.Glove

if embed_type == EMBED_TYPE.Glove:
    embed_dir = '../glove.840B.300d.txt'
elif embed_type == EMBED_TYPE.FastText:
    embed_dir = '../wiki-news-300d-1M-subword.vec' 


class BilmEmbed(Bilm):

    def __init__(self):
        super().__init__()
        self.embed = load_embedding(embed_dir)
        self.softmax_w = np.array(list(self.embed.values()))

    def _get_top_words_dist(self, state, cutoff):
        log_probs = np.matmul(state, self.softmax_w)
        top_k_log_probs = np.argpartition(-log_probs, cutoff)[: cutoff]
        top_k_log_probs_vals = log_probs[top_k_log_probs]
        e_x = np.exp(top_k_log_probs_vals - np.max(top_k_log_probs_vals))
        probs = e_x / e_x.sum(axis=0)
        return top_k_log_probs, probs

    def _get_embed(self, word:str)->List[float]:
        ''' get embed '''
        if word in self.embed:
            return self.embed[word]
        else:
            return np.zeros(300)

    def _embed_sentences(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]], disable_symmetric_patterns: bool) -> Tuple[List, List]:
        inst_id_sent_tuples = list(inst_id_to_sentence.items())
        target = inst_id_sent_tuples[0][0].rsplit('.', 1)[0]
        to_embed = []

        if disable_symmetric_patterns:
            for _, (tokens, target_idx) in inst_id_sent_tuples:
                forward = tokens[:target_idx]
                backward = tokens[target_idx + 1:]
                if not forward:
                    forward = ['.']
                if not backward:
                    backward = ['.']
                to_embed.append(forward)
                to_embed.append(backward)
        else:
            for _, (tokens, target_idx) in inst_id_sent_tuples:
                to_embed.append(tokens[:target_idx + 1] + ['and'])
                to_embed.append(['and'] + tokens[target_idx:])
        to_embed = [sum([self._get_embed(jj) for jj in ii]) for ii in to_embed]

        return inst_id_sent_tuples, to_embed

    def predict_sent_substitute_representatives(self, inst_id_to_sentence: Dict[str, Tuple[List[str], int]],
                                                n_represent: int,
                                                n_samples_side: int, disable_symmetric_patterns: bool,
                                                disable_lemmatiziation: bool, prediction_cutoff: int) \
            -> Dict[str, List[Dict[str, int]]]:
        """
        a representative is a dictionary made out of samples from both sides of the BiLM, predicting substitutes
        for a contextualized token.
        an example might look like:
        {'forward_jump':2,'backward_leap':1, 'backward_climb':1} (n_samples_side=2)
        we return a list of n_representatives of those

        :param inst_id_to_sentence: dictionary instance_id -> (sentence tokens list, target word index in tokens)
        :param n_represent: number of representatives
        :param n_samples_side: number of samples to draw from each side
        :param disable_symmetric_patterns: if true words are predicted from context only
        :param disable_lemmatiziation: if true predictions are not lemmatized
        :param prediction_cutoff: only top prediction_cutoff LM prediction are considered
        :return: map from instance id to list of representatives
        """
        inst_id_sent_tuples, embedded = self._embed_sentences(
            inst_id_to_sentence, disable_symmetric_patterns)
        lemma = inst_id_sent_tuples[0][0].split('.')[0]

        vocabulary_used = self.elmo_word_vocab if disable_lemmatiziation else self.elmo_word_vocab_lemmatized

        results = {}
        for i in range(len(inst_id_sent_tuples)):
            inst_id, (tokens, target_idx) = inst_id_sent_tuples[i]
            target_word_lower = tokens[target_idx].lower()

            forward_out_em = embedded[i * 2]
            backward_out_em = embedded[i * 2 + 1]

            forward_idxs, forward_dist = self._get_top_words_dist(forward_out_em, prediction_cutoff)
            backward_idxs, backward_dist = self._get_top_words_dist(backward_out_em, prediction_cutoff)

            forward_samples = []
            while len(forward_samples) < n_represent * n_samples_side:
                new_samples = list(np.random.choice(forward_idxs, n_represent * n_samples_side * 2, p=forward_dist))
                new_samples = [vocabulary_used[x] for x in new_samples if
                               vocabulary_used[x].lower() != lemma and vocabulary_used[x].lower() != target_word_lower]
                forward_samples += new_samples

            backward_samples = []
            while len(backward_samples) < n_represent * n_samples_side:
                new_samples = list(
                    np.random.choice(backward_idxs, n_represent * n_samples_side * 2, p=backward_dist))
                new_samples = [vocabulary_used[x] for x in new_samples if
                               vocabulary_used[x].lower() != lemma and vocabulary_used[x].lower() != target_word_lower]
                backward_samples += new_samples
            logging.info('some forward samples: %s' % [x for x in forward_samples[:5]])
            logging.info('some backward samples: %s' % [x for x in backward_samples[:5]])
            representatives = []
            for _ in range(n_represent):
                representative = dict()
                for _ in range(n_samples_side):
                    for sample_src in forward_samples, backward_samples:
                        sample_word = sample_src.pop()
                        representative[sample_word] = representative.get(
                            sample_word, 0) + 1
                representatives.append(representative)
            results[inst_id] = representatives
        return results
