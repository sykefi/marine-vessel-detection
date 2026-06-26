#!/bin/bash

annotation_path=dataset/annotations/
mosaic_path=dataset/rgb_mosaics/
mask_path=dataset/masks/
mkdir $mask_path

train_tiles=("34VEM" "34WFT" "35VLG" "34VEN" "34VER")

for tile in ${train_tiles[@]}; do
    mosaics=$(ls $mosaic_path/$tile/*.tif)
    for mosaic in ${mosaics[@]}; do
        [[ "$mosaic" =~ _([0-9]{8})T[0-9]{6}_ ]] && layer="${BASH_REMATCH[1]}"
        python scripts/create_sam3_masks.py \
            $annotation_path/${tile}.gpkg \
            $mosaic \
            $mask_path \
            --use_cuda \
            --gpkg_layer $layer
    done
done