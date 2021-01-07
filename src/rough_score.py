# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:27:48 2021

@author: NguyenSon
"""

import numpy as np
from rouge_score import rouge_scorer
from datetime import datetime


def rouge_scoring(df, raw_col, machine_col):
    '''
    Rouge Scoring on Rouge-1, Rouge-2, Rouge-L

    Parameters
    ----------
    df : TYPE Dataframe
        DESCRIPTION. result dataframe. Expected to have original summary column and summary by machine
    raw_col : TYPE string
        DESCRIPTION. Name of raw human summary
    machine_col : TYPE string
        DESCRIPTION. Name of machine generate summary

    Returns
    -------
    rouge_score : TYPE dict
        DESCRIPTION. Dictionary result

    '''
    scr_type = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(scr_type, use_stemmer=True)

    rouge_score = {'rouge1':{'precision':[],
                             'recall':[],
                             'fmeasure': []},
                   'rouge2':{'precision':[],
                             'recall':[],
                             'fmeasure': []},
                   'rougeL':{'precision':[],
                             'recall':[],
                             'fmeasure': []}}
    
    for i, r in df.iterrows():
        score_all = scorer.score(r[raw_col],r[machine_col])
        for t in scr_type:
                rouge_score[t]['precision'].append(score_all[t].precision)
                rouge_score[t]['recall'].append(score_all[t].recall)
                rouge_score[t]['fmeasure'].append(score_all[t].fmeasure)
    for t in scr_type:
        for k, v in rouge_score[t].items():
            rouge_score[t][k] = np.mean(np.array(v))
            print('{} ==== {}-{}: {}'.format(datetime.now(),t.upper(),k.upper(), np.mean(np.array(v))))
    return rouge_score

# result = pd.read_csv('./data/predict_0.csv')
# rouge_score = scoring(result, 'originSum', 'machineSum')