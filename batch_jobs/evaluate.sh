#!/bin/bash

model_dir=runs
models=$( cd $model_dir && echo */ ) 
mosaic_path=dataset/rgb_mosaics
annotation_path=dataset/masks
result_dir=preds
mkdir $result_dir
test_tiles=("34VEN" "34VER")

for model in ${models[@]}; do

    # Collate validation metrics and find the optimal confidence based on validation F1
    echo Validating $model
    python scripts/get_val_metrics.py \
           $model_dir/$model \
           yolo_dataset/fold_1.yaml

    conf=$( awk -F',' 'NR==2 {print $7}' $model_dir/$model/val_results_best.csv )
    echo Processing $model, using confidence threshold of $conf
    mkdir $result_dir/$model
    for tile in ${test_tiles[@]}; do
        echo Processing tile $tile
        mkdir $result_dir/$model/$tile
        mosaics=$(ls $mosaic_path/$tile/*.tif)
        for mosaic in ${mosaics[@]}; do
            echo Processing tile $mosaic
            python scripts/predict_tile.py \
                $model_dir/$model/weights/best.pt \
                $mosaic \
                $result_dir/$model/$tile \
                --use_cuda \
                --conf $conf \
                --postproc \
                --preset $tile \
                --image_size 640 \
                --slice_size 320 \
                --keep_fps \
                --half
        done
    done

    # Clip 34VEN predictions
    preds_to_clip=$(ls $result_dir/$model/34VEN/)
    echo Clipping 34VEN preds to relevant area
    for pred in ${preds_to_clip[@]}; do
        tempfile=$result_dir/$model/34VEN/temp.geojson
        ogr2ogr -clipsrc $mosaic_path/ven_area.geojson \
            $tempfile \
            $result_dir/$model/34VEN/$pred
        mv $tempfile $result_dir/$model/34VEN/$pred
    done

    echo Evaluating $model
    python scripts/evaluate.py \
           $annotation_path \
           $result_dir/$model/ \
           $model_dir/$model/ \
           --conf_thr 0.001

    python scripts/evaluate.py \
           $annotation_path \
           $result_dir/$model/ \
           $model_dir/$model/ \
           --conf_thr 0.001 --filter_preds
done

# Make collated test and validation reports

python scripts/make_validation_report.py \
       $model_dir \
       $model_dir/val_results_best.csv \
       --model_type best

python scripts/make_validation_report.py \
       $model_dir \
       $model_dir/val_results_last.csv \
       --model_type last

python scripts/make_test_report.py \
       $model_dir all_results \
       $model_dir/test_results_raw.csv

python scripts/make_test_report.py \
       $model_dir all_results_filtered \
       $model_dir/test_results_filtered.csv