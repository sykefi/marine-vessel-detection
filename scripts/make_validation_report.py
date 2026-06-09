import pandas as pd
from pathlib import Path
import os
from fastcore.script import *
from itertools import product

@call_parse
def make_result_report(
    model_dir:Path, # Path containing the trained models
    outfile:Path, # Path of the resulting csv file
    model_type:str='best' # collate results for best or last model
):
    outdf = None

    model_arcs = [f for f in os.listdir(model_dir) if os.path.isdir(model_dir/f)]

    for model in (model_arcs):
        if not os.path.exists(model_dir/model/f'val_results_{model_type}.csv'): continue
        res = pd.read_csv(model_dir/model/f'val_results_{model_type}.csv')
        res['model_type'] = model.split('_')[0]
        res['data_type'] = model.split('_')[-1]
        res['pred_type'] = 'one2many'
        if outdf is None: outdf = res
        else: outdf = pd.concat([outdf, res])

    for model in (model_arcs):
        if not os.path.exists(model_dir/model/f'val_results_{model_type}_end2end.csv'): continue
        res = pd.read_csv(model_dir/model/f'val_results_{model_type}_end2end.csv')
        res['model_type'] = model.split('_')[0]
        res['data_type'] = model.split('_')[-1]
        res['pred_type'] = 'one2one'
        if outdf is None: outdf = res
        else: outdf = pd.concat([outdf, res])

    outdf.reset_index(inplace=True, drop=True)
    outdf = outdf[['model_type', 'data_type', 'pred_type', 'precision', 'recall', 'f1',
                   'ap50', 'ap75', 'ap', 'conf', 'speed(ms)']]
    outdf.sort_values(by='f1', ascending=False, inplace=True)
    outdf.to_csv(outfile, index=False)