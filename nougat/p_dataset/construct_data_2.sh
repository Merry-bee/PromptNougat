#!/bin/bash

# 切换到项目根目录
cd /mnt/workspace/sunyu/nougat
conda activate base

# cpu数：35

# 设置并发进程数量
concurrent_limit=40
# 每个进程处理文件数量
num_fold=1000
# 设置文件起始idx,处理文件范围：810,000~810,000+40*1,000=810,000~850,000
start_from=810000
# arxiv_all_files_idx
save_fold='arxiv_all_files3'

for ((i=0;i<$concurrent_limit;i++)); do

    start_idx="$((start_from+i*num_fold))"
    sem -j $concurrent_limit nohup python nougat/p_dataset/tex+color.py "$start_idx" "$num_fold" "$save_fold" > log/construct_data_2.log 2>&1 &

done