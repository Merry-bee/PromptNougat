python -m pdb test_prompt.py \
--model_path result/nougat/20231207_183828 \
-d data/train_data/validation_lorem1.jsonl \
--save_path output/tmp \
--split train \
--batch_size 1 \
--ckpt_path result/nougat/20231207_183828/epoch=201-step=56358.ckpt