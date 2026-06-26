import numpy as np
import torch
from tqdm import tqdm
from shapely.geometry import shape
from shapely import affinity
import rasterio as rio
import geopandas as gpd
from pathlib import Path

import skimage as ski

import sam3
import os
SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from geo2ml.data.coordinates import gdf_to_px, georegister_px_df

from fastcore.script import *
from rasterio.features import shapes
from PIL import Image

@call_parse
def create_sam3_masks(
    boxes:Path, # Path to annotations
    mosaic:Path, # Path to mosaic
    outpath:Path, # Directory to save the results
    use_cuda:bool, # Whether to use CUDA for predictions
    gpkg_layer:str=None, # Spefify GPKG layer to use
):
    outpath = Path(outpath)
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'

    model = build_sam3_image_model(bpe_path=f"{SAM3_ROOT}/assets/bpe_simple_vocab_16e6.txt.gz", 
                                   device=device,
                                   enable_inst_interactivity=True).to(torch.float32)
    processor = Sam3Processor(model)
    if boxes.suffix == '.geojson':
        gdf = gpd.read_file(boxes)
    elif boxes.suffix == '.gpkg':
        gdf = gpd.read_file(boxes, layer=gpkg_layer)
    px_gdf = gdf_to_px(gdf, mosaic)
    polys = []

    dx = 70
    dy = 70

    with rio.open(mosaic) as src:
        im = np.moveaxis(src.read(), 0, 2)

    for row in tqdm(px_gdf.itertuples(), total=len(px_gdf)):
        x = int(row.geometry.centroid.x) # centroid coords
        y = int(row.geometry.centroid.y) # centroid coords
        
        miny = max(0, y-dy)
        maxy = min(10980, y+dy)
        minx = max(0,x-dx)
        maxx = min(10980, x+dx)
        b = row.geometry.exterior.bounds # Bounding box in pixel coordinates
        bbox = np.array([b[0]-minx, b[1]-miny, b[2]-minx, b[3]-miny]) # Scale bounding box
        tempim = im[miny:maxy,minx:maxx] # crop

        # uncomment if your gpu supports bfloat16
        #with torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(Image.fromarray(tempim))

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=True)
        
        best_ix = np.argmax(scores)
        best_mask = masks[best_ix]
        if masks.sum() == 0: continue

        # 1 pixel closing should fix the gaps
        best_mask = ski.morphology.closing(best_mask, footprint=ski.morphology.disk(1))
        poly = shape(list(shapes(best_mask))[0][0])
        poly = affinity.translate(poly, xoff=minx, yoff=miny)
        polys.append(poly)

    poly_gdf = georegister_px_df(
        gpd.GeoDataFrame({'geometry': polys}), 
        mosaic
    )

    outfn = f"{boxes.stem}_masks{boxes.suffix}" 

    if gpkg_layer is None:
        poly_gdf.to_file(outpath/outfn)
    else: 
        poly_gdf.to_file(outpath/outfn, layer=gpkg_layer)
