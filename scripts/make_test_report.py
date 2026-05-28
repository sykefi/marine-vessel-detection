import pandas as pd 
from pathlib import Path 
import os
from fastcore.script import *
from itertools import product

@call_parse
def make_test_report(
    metric_dir:Path, # Path containing the metrics for each model
    eval_fname:str, # name for the files to search
    outfile:Path, # Where to save the resulting csv file
):
    outdf = None
    model_arcs = [f for f in os.listdir(metric_dir) if os.path.isdir(metric_dir/f)]
    for model in model_arcs:
        if not os.path.exists(metric_dir/model/f'{eval_fname}.json'): continue
        res = pd.read_json(metric_dir/model/f'{eval_fname}.json', orient='index').T
        res.rename(columns={'mAP50': 'ap50', 'mAP50-95': 'ap'}, inplace=True)
        res['model_type'] = model.split('_')[0]
        res['train_type'] = model.split('_')[-1]
        res['f1'] = 2 * (res.precision * res.recall)/(res.precision + res.recall)
        if outdf is None: outdf = res
        else: outdf = pd.concat([outdf, res])
    
    outdf.reset_index(inplace=True, drop=True)
    outdf = outdf[['model_type', 'train_type', 'precision', 'recall', 'f1',
                   'ap50', 'ap', 'tp', 'fp', 'fn']]
    outdf.to_csv(outfile, index=False)