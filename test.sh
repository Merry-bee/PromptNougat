python -m pdb test_prompt.py \
--model_path result/nougat/20231202_090540 \
-d data/train_data/validation_lorem1.jsonl \
--save_path output/tmp \
--split validation \
--batch_size 2 \
--ckpt_path result/nougat/20231202_090540/epoch=5-step=24.ckpt