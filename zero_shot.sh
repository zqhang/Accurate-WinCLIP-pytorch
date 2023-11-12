#!/bin/bash

few_shots=(0)

for few_num in "${!few_shots[@]}";do
## train on the VisA dataset
    base_dir=winclip_mvtec
    save_dir=./exps_${base_dir}/mvtecvit_large_14_518/

    CUDA_VISIBLE_DEVICES=3 python reproduce_WinCLIP.py --dataset mvtec \
    --data_path /remote-home/iot_zhouqihang/data/mvdataset --save_path ./results/mvtec_${base_dir}/few_shot_${few_shots[few_num]} \
    --model ViT-B-16-plus-240 --pretrained openai --k_shot ${few_shots[few_num]} --image_size 240 
    wait
done


for few_num in "${!few_shots[@]}";do
## train on the VisA dataset
    base_dir=winclip_visa
    save_dir=./exps_${base_dir}/mvtecvit_large_14_518/


    CUDA_VISIBLE_DEVICES=3 python reproduce_WinCLIP.py --dataset visa \
    --data_path /remote-home/iot_zhouqihang/data/Visa --save_path ./results/mvtec_${base_dir}/few_shot_${few_shots[few_num]} \
    --model ViT-B-16-plus-240 --pretrained openai --k_shot ${few_shots[few_num]} --image_size 240 
    wait
done
