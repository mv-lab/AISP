#!/bin/bash

echo "Start to train the model...."

name="p20"
dataroot="/home/work/ssd1/hagongda/lxy/data_p20"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --dataset_name p20patch   --model  s7naf2   --name $name         --gcm_coord False  \
    --pre_ispnet_coord False  --niter 200     --lr_policy cosine    --save_imgs False \
    --batch_size 4          --print_freq 300    --calc_metrics True  --lr 3e-4   -j 8 \
    --weight_decay 0.001 --patch_size 704 --load_iter 99 --load_optimizers True \
    --dataroot $dataroot | tee $LOG   
    
    
