#!/bin/bash

echo "Start to train the model...."

name="s7_3000"
dataroot="/mnt/disk10T/AIM2022/data-s7-before"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

# You can set "--model zrrganjoint" to train LiteISPGAN.

python train.py \
    --dataset_name s7align    --model  s7smooth   --name $name         --gcm_coord False  \
    --pre_ispnet_coord False  --niter 1800        --lr_policy cosine    --save_imgs False \
    --batch_size 6          --print_freq 300    --calc_metrics True  --lr 3e-4   -j 24  \
    --weight_decay 0.01  \
    --dataroot $dataroot | tee $LOG   
    
    
