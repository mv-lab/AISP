#!/bin/bash
echo "Start to test the model...."

name="p20"
dataroot="/home/work/ssd1/hagongda/lxy/data_p20/test_full"
save_path='./submission'


python test_full.py \
--model  naf  --name $name      --dataset_name p20patch  --pre_ispnet_coord False  --gcm_coord False \
--load_iter  82  --batch_size 1  --gpu_ids -1  --save_imgs True  --calc_metrics True  --visual_full_imgs False  -j 3 \
--dataroot $dataroot --save_path $save_path


