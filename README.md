# Detecting marine vessels from Sentinel-2 images with YOLO models

Identifying where maritime activities take place, and quantifying their potential impact on marine biodiversity, is important for sustainable management of marine areas, spatial planning and marine conservation. Detection and monitoring of small vessels, such as pleasure crafts, has been challenging due to limited data availability with adequate temporal and spatial resolution. This repository contains an approach to quantify maritime traffic with YOLO object detection models.

## Installation

Install the required environment with

```bash
conda env create -f environment.yml
conda activate ship-env
```

Environment contains all required geospatial packages, `omnicloudmask` for cloud mask creation `ultralytics` for detection tasks and `geo2ml` for converting geospatial data to ultralytics compatible format and vice versa.

### Optional: sam3

[sam3](https://github.com/facebookresearch/sam3) can optionally be used to create polygon masks from the bounding box annotations. Creating the masks is also beneficial for bounding box detections, as creating the boxes with the `bounds` attribute of a polygon might be more accurate than human-drawn one. 

sam3 can be installed with

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# Optionally install optional dependencies for faster inference
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git
```

Finally fix `sam3/model_builder.py` with [these changes](https://github.com/facebookresearch/sam3/pull/537/changes).

For getting access to sam3 checkpoints, follow the instructions on sam3 repository.

## Data

### Satellite data

Lists of Sentinel-2 mosaics matching the annotations can be found in `dataset/train_products.txt` and `dataset/test_products.txt`. Easiest way to access them is via [CDSE OData API](https://documentation.dataspace.copernicus.eu/APIs/OData.html), using `scripts/download_mosaics.py`. CDSE services require authentication, so after creating the account, save the credentials on a json-file with structure in some safe location:

```json
{
    "grant_type": "password",
    "username": "your_username_here",
    "password": "your_password_here",
    "client_id": "cdse-public"
}
```

Then you can download the data with 

```bash
python scripts/download_mosaics.py \
       train_products.txt \
       secret/creds.json \
       temp_download_path \
       mosaic_outpath/ \
```

If you want to create also the cloud masks with `omnicloudmask`, add flags `--make_cloud_masks --cloudmask_outpath cloudmask_outpath` to the previous command.

### Vessel annotations

Manually annotated vessel bounding box data can be accessed from Zenodo: [10.5281/senodo.10046341](https://zenodo.org/records/15019034). Each geopackage corresponds to a single Sentinel-2 tile, and each layer in a geopackage corresponds to a single date.

As tiles 34VEM and 34VEN have around 20km overlap with each other, the overlapping area is annotated only for 34VEM.

### Pretrained models

Model checkpoints are available on 🤗 as soon as the training and evaluation is complete.

The models are trained using Sentinel-2 L1C True-color images, processed with processing baseline `N0500` or newer.

## Pipeline 

For all scripts, running `python scripts/script.py -h` shows the help for that step.

### Optional: using sam3 to create instance masks

While using the original hand-drawn bounding boxes provides satisfactory results, enhancing the boxes with sam3 is suggested. Running `./batch_jobs/create_sam3_masks.sh` processes all images so that it runs through all bounding boxes, extracts a 140x140 pixel image around it, selects the mask with highest score and runs morphological closing with 1 pixel disk footprint to unify the mask, finally saving the results as polygons.

### ultralytics dataset creation

`scripts/make_yolo_data.py` converts the geopackages and mosaics into `ultralytics` compatible datasets.

Default command for dataset creation is 

```bash
python make_yolo_data.py path_to_annotations path_to_mosaics outpath --tilesize 320 --overlap --n_folds 5
```

The default train-val -split for train tiles is to create a 5x5 grid over each of the tiles, and split this grid into five subsets diagonally. Default option is to use `fold_1` for validation and other data for training. 

For training the models, we used 320x320 image patches with zero overlap, which were then upsampled to 640x640 images by `ultralytics`. 

### Training

After creating the models, training can be done with `src/train_model.py`

### Validation and evaluation

The main reason for manual validation is to get the confidence score with the best F1-score to use for inference. `scripts/get_val_metrics.py model_dir yolo_dataset/fold_1.yaml` can be used to get validation metrics, optionally filtering only relevant predictions with `--filter_preds`. 

Evaluation is done based on Sentinel-2 tiles 34VEN and 34VER, where each timestep is considered as a single image. Steps to evaluate the models this way are

1. Run the predictions for each train tile
2. Clip predictions from 34VEN to only contain the non-overlapping area with 34VEM
3. Run `scripts/evaluate.py`, which outputs results for each tile individually, as well as overall results. 

### Using the models for new areas

1. Download the L1C product 
2. Convert the product to L1C True Color Image
3. Run `scripts/predict_tile.py` (check `python scripts/predict_tile.py -h` for help)

Optional: For Finnish areas, it is recommended to use external datasets for cleaning stationary targets, predictions on land, large rocks and beacons from the predictions. Most of the cleaning is done based on [Topographic Database](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/topographic-database), whereas beacons and other can be accessed with data from [Finnish Transport Infrastructure Agency](https://vayla.fi/en/transport-network/data/open-data), layer `Aids to navigation`. 

Unfortunately, layers for fisheries must be created manually.

Paths for these external data must be specified to `src/data_paths.py`. Modify the examples according to your needs.

## Citation

Original workflow was developed in

> Mäyrä, J., Virtanen, E. A., Jokinen, A.-P., Koskikala, J., Väkevä, S., & Attila, J. (2025). Mapping recreational marine traffic from sentinel-2 imagery using Yolo Object Detection Models. Remote Sensing of Environment, 326, 114791. https://doi.org/10.1016/j.rse.2025.114791

```bibtex
@article{mayraMappingRecreational2025,
title = {Mapping recreational marine traffic from Sentinel-2 imagery using YOLO object detection models},
journal = {Remote Sensing of Environment},
volume = {326},
pages = {114791},
year = {2025},
issn = {0034-4257},
doi = {https://doi.org/10.1016/j.rse.2025.114791},
url = {https://www.sciencedirect.com/science/article/pii/S0034425725001956},
author = {Janne Mäyrä and Elina A. Virtanen and Ari-Pekka Jokinen and Joni Koskikala and Sakari Väkevä and Jenni Attila},
keywords = {Marine vessel detection, Object detection, Satellite imagery, Deep learning, Human pressures},
}
```

## Acknowledgments

This project was supported by Enhancing the marine and coastal biodiversity of the Baltic Sea in Finland and promoting the sustainable use of marine resources (LIFE-IP BIODIVERSEA (LIFE20 IPE/FI/000020)).

The authors wish to acknowledge CSC – IT Center for Science, Finland, for computational resources.