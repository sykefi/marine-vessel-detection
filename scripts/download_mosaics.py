from cdse_odata import FileDownloader
from multiprocessing import Pool
import os
from fastcore.script import *
from shutil import rmtree
from itertools import cycle
import time
from omnicloudmask import predict_from_load_func, load_s2
from pathlib import Path
import torch

import numpy as np
import glob
import rasterio as rio
from fastcore.script import *
import os

def make_mosaic(input, outfile):
    SATURATION_VALUE = 3558 

    xmlroot = glob.glob(os.path.join(input, 'MTD_MSIL1C.xml'))[0]

    print('Reading data')
    with rio.open(xmlroot, 'r') as src:
        data_10m = src.subdatasets[0]

    # Derive offset_value depending on the baseline
    baseline_ver = int(input.split('_')[-4][2:])
    if baseline_ver >= 400: RADIO_ADD_OFFSET = -1000
    else: RADIO_ADD_OFFSET = 0

    with rio.open(data_10m, 'r') as src:
        B04 = (src.read(1).astype(int) + RADIO_ADD_OFFSET)/(SATURATION_VALUE+RADIO_ADD_OFFSET)
        B03 = (src.read(2).astype(int) + RADIO_ADD_OFFSET)/(SATURATION_VALUE+RADIO_ADD_OFFSET)
        B02 = (src.read(3).astype(int) + RADIO_ADD_OFFSET)/(SATURATION_VALUE+RADIO_ADD_OFFSET)
        prof = src.profile

    r = np.clip(255*B04, 0, 255).astype(np.uint8)
    g = np.clip(255*B03, 0, 255).astype(np.uint8)
    b = np.clip(255*B02, 0, 255).astype(np.uint8)

    prof.update(
        count=3,
        dtype='uint8',
        driver='GTiff', 
        compress='lzw',
        predictor=2,
        BIGTIFF='YES'
    )

    print('Writing data')
    with rio.open(outfile, 'w', **prof) as dst:
        dst.write(np.stack([r, g, b]))


def run_chain(product_name, downloader, dl_path, mosaic_outpath,
              make_cloud_masks, cloudmask_outpath):
    outfile = f'{product_name.replace("SAFE", "tif")}'
    num_downloads = 0
    if not os.path.exists(f'{mosaic_outpath}/{outfile}'):
        while num_downloads < 10:
            try:
                downloader.refresh_token()
                downloader.query_product_by_name(name=product_name)
                downloader.download_latest_response(target_path=dl_path)
                make_mosaic(input=f'{dl_path}/{product_name}', 
                            outfile=f'{mosaic_outpath}/{outfile}')
            except:
                num_downloads += 1
                print(f'Error while processing {product_name}, retry {num_downloads}/10')
                time.sleep(30) # Wait for 30 seconds
            else:
                if make_cloud_masks:
                    if torch.cuda.is_available():
                        _ = predict_from_load_func([f'{dl_path}/{product_name}'], load_s2,
                                                    inference_device='cuda',
                                                    batch_size=4, inference_dtype='fp16',
                                                    output_dir=cloudmask_outpath)
                    else:
                        _ = predict_from_load_func([f'{dl_path}/{product_name}'], load_s2,
                                                    inference_device='cpu',
                                                    batch_size=4, inference_dtype='fp32',
                                                    output_dir=cloudmask_outpath)
                break
        if num_downloads == 10:
            print(f'Failed to access {product_name}')
        rmtree(f'{dl_path}/{product_name}')
    else: print(f'{outfile} exist, skipping...')
    return

@call_parse
def download_and_convert(
    product_name_file:str, # Path to txt file containing the product ids
    creds_file:str, # Path to credential files
    dl_path:str, # tmp path to download to
    mosaic_outpath:Path, # Path to output mosaic folder
    make_cloud_masks:bool, # Whether to make cloud masks, saved in
    cloudmask_outpath:Path = None, # Path for cloud masks
):
    "Wrapper to both download and convert data with multiprocessing"

    if make_cloud_masks and cloudmask_outpath == None:
        print('Specify the path for cloud_masks')

    with open(product_name_file) as f:
        lines = [line.rstrip() for line in f]

    downloaders = [FileDownloader(username=None, password=None, creds_file=creds_file) for _ in range(4)]

    inps = [(product_name, 
             downloader, 
             dl_path, 
             mosaic_outpath,
             make_cloud_masks,
             cloudmask_outpath) 
            for product_name, downloader in zip(lines, cycle(downloaders))]
    print(f'{len(inps)} products to download')
    with Pool(4) as pool:
        pool.starmap(run_chain, inps)
