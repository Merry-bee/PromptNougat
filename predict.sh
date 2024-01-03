python -m pdb predict.py data/tmp/trainset/1010.1542.pdf \
--out output/greedy_search \
--checkpoint result/nougat/20231227_195400 \
--batchnum 8 \
--batchsize 1 \
--cuda "cuda:0" \
--recompute \
--return_attention True \
--ckpt_path result/nougat/20231227_195400/epoch=5-step=129072.ckpt \
