# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-18 22:30:39
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-19 14:15:13
import sys
from typing import Dict

import numpy as np

from spwsi.semeval_utils import evaluate_labeling, generate_sem_eval_2013


def evaluateFromFile(input_path: str):
    ''' evaluation from file '''
    with open(input_path, 'r') as f:
        test = [ii.strip().split() for ii in f.readlines()]
    data = {ii[1]: {jj.split('/')[0]: float(jj.split('/')[1]) for jj in ii[2:]} for ii in test}
    handleScore(data, input_path)


def handleScore(data: Dict[str, Dict[str, float]], evaluationId: str, outPath: str = None) -> Dict[str, Dict[str, float]]:
    ''' handle score '''
    scores = evaluate_labeling('./resources/SemEval-2013-Task-13-test-data', data, outPath)
    # print(scores['all'])
    jaccard = scores['all']['jaccard-index']
    pos = scores['all']['pos-tau']
    WNDC = scores['all']['WNDC']
    fnmi = scores['all']['FNMI']
    fbc = scores['all']['FBC']
    msg = 'Result: jaccard|pos  |WNDC |FNMI|FBC  |AVG  |\n          '
    for ii in scores['all'].values():
        msg += '{:.2f}|'.format(ii * 100)
    msg += '{:.2f}|'.format(np.sqrt(fnmi * fbc) * 100)
    with open('debug/evaluation.log', 'a') as f:
        f.write('{}: {}\n'.format(evaluationId, msg.split('\n')[1].strip()))
    print(msg)
    return scores


if __name__ == '__main__':

    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    if input_path is None:
        raise Exception('missing inputPath')
    else:
        evaluateFromFile(input_path)
