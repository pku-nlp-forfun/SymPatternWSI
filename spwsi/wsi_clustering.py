from typing import Dict, List
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.pipeline import make_pipeline
from collections import Counter
import logging
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch
import numpy as np

def load_embedding(data_path:str) -> Dict[str, List[float]]:
    ''' load embedding '''
    with open(data_path) as f:
        origin_embed = [ii.strip() for ii in f.readlines()]
    embed = {ii.split(' ')[0]: np.array(ii.split(' ')[1:]).astype(float) for ii in origin_embed}
    return embed

def get_embed(word:str)->List[float]:
    if word in gloved_embed:
        return gloved_embed[word]
    else:
        return np.zeros(300)

bert_dir = '../bert'
bert = BertModel.from_pretrained(bert_dir)
tokenizer = BertTokenizer.from_pretrained(f'{bert_dir}/uncased_L-24_H-1024_A-16/vocab.txt')
gloved_path = '../wiki-news-300d-1M-subword.vec'
gloved_embed = load_embedding(gloved_path)


def cluster_inst_ids_representatives(inst_ids_to_representatives: Dict[str, List[Dict[str, int]]],
                                     n_clusters: int, disable_tfidf: bool, embedType:int=0) -> Dict[str, Dict[str, int]]:
    """
    preforms agglomerative clustering on representatives of one SemEval target
    :param inst_ids_to_representatives: map from SemEval instance id to list of representatives
    :param n_clusters: fixed number of clusters to use
    :param disable_tfidf: disable tfidf processing of feature words
    :return: map from SemEval instance id to soft membership of clusters and their weight
    """
    inst_ids_ordered = list(inst_ids_to_representatives.keys())
    lemma = inst_ids_ordered[0].rsplit('.', 1)[0]
    logging.info('clustering lemma %s' % lemma)
    representatives = [y for x in inst_ids_ordered for y in inst_ids_to_representatives[x]]
    n_represent = len(representatives) // len(inst_ids_ordered)
    to_pipeline = [DictVectorizer()]
    if embedType == 1:
        waitSentence = [' '.join(ii) for ii in representatives]
        transformed = []
        for ii in waitSentence:
            ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ii))])
            transformed.append(bert(ids, output_all_encoded_layers=False)[-1][0].detach().numpy())
    elif embedType == 2:
        transformed = [sum([get_embed(jj) for jj in ii]) for ii in representatives]
    else:
        if not disable_tfidf:
            to_pipeline.append(TfidfTransformer())
        data_transformer = make_pipeline(*to_pipeline)
        transformed = data_transformer.fit_transform(representatives).todense()
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average', affinity='cosine')
    clustering.fit(transformed)
    senses = {}
    for i, inst_id in enumerate(inst_ids_ordered):
        inst_id_clusters = Counter(clustering.labels_[i * n_represent: (i + 1) * n_represent])
        senses[inst_id] = dict([('%s.sense.%d' % (lemma, k), v) for (k, v) in inst_id_clusters.most_common()])
    return senses
