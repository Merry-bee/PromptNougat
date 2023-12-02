python -m pdb predict.py data/quantum/10.1088/0305-4470%2F37%2F38%2FL01.pdf \
--out output/greedy_search \
--checkpoint checkpoints/0.1.0-small \
--batchnum 8 \
--batchsize 1 \
--cuda "cuda:0" \
--recompute \
--return_attention True
# --ckpt_path result/nougat/20231122_155242_auto/epoch=14-step=5010.ckpt \
