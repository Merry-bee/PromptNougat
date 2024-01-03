#!/bin/bash

# train0.jsonl:0-34w = 9/34 = 0.26
# train1.jsonl:34-55w = 4/21 = 0.1
# train2.jsonl:55-76w = 0.8/21 = 0.03
# train3.jsonl:76-97w

# 切换到项目根目录
cd /mnt/workspace/sunyu/nougat
conda activate base

# cpu数：55

# 设置并发进程数量
concurrent_limit=50
# 每个进程处理文件数量
num_fold=1000
# 设置文件起始idx, 处理文件范围：760,000~760,000+50*1,000=760,000~810,000
start_from=760000
# arxiv_all_files_idx
save_fold='arxiv_all_files3'

for ((i=0;i<$concurrent_limit;i++)); do

    start_idx="$((start_from+i*num_fold))"
    sem -j $concurrent_limit nohup python nougat/p_dataset/tex+color.py "$start_idx" "$num_fold" "$save_fold"> log/construct_data_1.log 2>&1 &

done
