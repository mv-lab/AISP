#!/bin/bash
echo "Start to test the model...."

name="s7_1000"
dataroot="/mnt/disk10T/AIM2022/data-s7-full/test2"
save_path='./submission'


python test_full.py \
--model  s7smooth   --name $name   --dataset_name s7align  --pre_ispnet_coord False  --gcm_coord False \
--load_iter  484 --batch_size 1 --gpu_ids -1  --save_imgs True  --calc_metrics True  --visual_full_imgs False  -j 3 \
--dataroot $dataroot --save_path $save_path  

