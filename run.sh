#!/usr/bin/env sh
python main.py \
    -img_path Path_to_the_images \
    -list_path ./miniimagenet-csv \
    -shots 5 \
    -num_classes 5 \
    -trails 200 \
    -net Path_to_the_model \
    -n_vcs 200 \
    -extract VMFM \
    -refine Cosine \
    -least_cov 0.8 \
    -sigma 1.2 \
    -classifier Likelihood \
    -record Path_to_save_the_records 
