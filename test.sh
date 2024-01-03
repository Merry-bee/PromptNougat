python -m pdb test_prompt.py \
--model_path result/nougat/20231227_195400 \
--dataset data/arxiv_train_data/validation.jsonl \
--save_path output/tmp \
--split train \
--batch_size 1 \
--visualize False \
--ckpt_path result/nougat/20231227_195400/epoch=5-step=129072.ckpt