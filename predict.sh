python -m pdb predict.py data/tmp/trainset/1010.1542.pdf \
--out output/greedy_search \
--checkpoint result/nougat/20240106_092902 \
--batchnum 8 \
--batchsize 1 \
--cuda "cuda:0" \
--recompute \
--return_attention True \
--ckpt_path result/nougat/20240106_092902/epoch=25-step=49400.ckpt \
