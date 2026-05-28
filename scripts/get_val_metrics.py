from fastcore.script import *
import pandas as pd
from pathlib import Path
import numpy as np
import os
import tempfile

os.environ['YOLO_VERBOSE'] = "false"
from ultralytics import YOLO

# disable update check
import ultralytics
ultralytics.utils.ONLINE = False

@call_parse
def get_val_results(
    model_dir:Path, # Path to the directory containing the model files
    val_data_path:Path, # Path to validation dataset
    half:bool=False, # Half precision?
):
    """Validate the models, save validation results and optimal confidence threshold based on f1 to `model_dir`"""
    if not os.path.exists(model_dir/'weights/best.pt'): return
    m = YOLO(model_dir/'weights/best.pt')
    # Replace data dir
    m.args['data'] = val_data_path
    with tempfile.TemporaryDirectory() as sd:
        res = m.val(device='cuda', project=None, save_dir=sd, plots=False, half=half, verbose=False, 
                    save_json=False, save_txt=False, save_conf=False, single_cls=True, conf=0.001, batch=1,
                    compile=True, show_conf=False, show_labels=False, visualize=False)
    print(f'Best precision ({res.curves_results[2][1].max():.4f}) achieved with conf: {res.curves_results[2][0][res.curves_results[2][1].argmax()]:.4f}')
    print(f'Best recall ({res.curves_results[3][1].max():.4f}) achieved with conf: {res.curves_results[3][0][res.curves_results[3][1].argmax()]:.4f}')
    print(f'Best F1-score ({res.curves_results[1][1].max():.4f}) achieved with conf: {res.curves_results[1][0][res.curves_results[1][1].argmax()]:.4f}')

    outdict = {
        'precision': [res.box.mp],
        'recall': [res.box.mr],
        'f1': [res.box.f1[0]],
        'ap50': [res.box.map50],
        'ap75': [res.box.map75],
        'ap': [res.box.map],
        'conf': [np.round(res.curves_results[1][0][res.curves_results[1][1].argmax()], 3)],
        'speed(ms)': [np.round(res.speed['inference'], 4)]
    }

    df = pd.DataFrame(outdict)
    df.to_csv(model_dir/'val_results_best.csv', index=False)

    m = YOLO(model_dir/'weights/last.pt')
    # Replace data dir
    m.args['data'] = val_data_path
    with tempfile.TemporaryDirectory() as sd:
        res = m.val(device='cuda', project=None, save_dir=sd, plots=False, half=half, verbose=False, 
                    save_json=False, save_txt=False, save_conf=False, single_cls=True, conf=0.001, batch=1,
                    compile=True, show_conf=False, show_labels=False, visualize=False)
    print(f'Last precision ({res.curves_results[2][1].max():.4f}) achieved with conf: {res.curves_results[2][0][res.curves_results[2][1].argmax()]:.4f}')
    print(f'Last recall ({res.curves_results[3][1].max():.4f}) achieved with conf: {res.curves_results[3][0][res.curves_results[3][1].argmax()]:.4f}')
    print(f'Last F1-score ({res.curves_results[1][1].max():.4f}) achieved with conf: {res.curves_results[1][0][res.curves_results[1][1].argmax()]:.4f}')

    outdict = {
        'precision': [res.box.mp],
        'recall': [res.box.mr],
        'f1': [res.box.f1[0]],
        'ap50': [res.box.map50],
        'ap75': [res.box.map75],
        'ap': [res.box.map],
        'conf': [np.round(res.curves_results[1][0][res.curves_results[1][1].argmax()], 3)],
        'speed(ms)': [np.round(res.speed['inference'], 4)]
    }

    df = pd.DataFrame(outdict)
    df.to_csv(model_dir/'val_results_last.csv', index=False)