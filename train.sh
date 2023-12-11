python train_prompt.py --config config/train_PromptNougat.yaml > log/tmp.out 2>&1 &
python -m pdb train_prompt.py --config config/train_tmp.yaml --debug 