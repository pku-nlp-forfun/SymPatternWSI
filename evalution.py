# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-05-18 22:30:39
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-05-18 23:09:39
import sys
import numpy as np
from spwsi.semeval_utils import generate_sem_eval_2013, evaluate_labeling


def evaluate(input_path:str):
    ''' evaluation '''
    with open(input_path, 'r') as f:
        test = [ii.strip().split() for ii in f.readlines()]
    inst_id_to_sense = {ii[1]:{jj.split('/')[0]:float(jj.split('/')[1]) for jj in ii[2:]} for ii in test}
    scores = evaluate_labeling('./resources/SemEval-2013-Task-13-test-data', inst_id_to_sense)
    # print(scores)
    fnmi = scores['all']['FNMI']
    fbc = scores['all']['FBC']
    msg = 'results FNMI %.2f FBC %.2f AVG %.2f' % (fnmi * 100, fbc * 100, np.sqrt(fnmi * fbc) * 100)
    print(msg)


if __name__ == '__main__':

    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    if input_path is None:
        raise Exception('missing inputPath')
    else:
        evaluate(input_path)