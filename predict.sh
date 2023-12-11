python -m pdb predict.py data/lorempdf/000/latex.pdf \
--out output/greedy_search \
--checkpoint result/nougat/20231207_183828 \
--batchnum 8 \
--batchsize 1 \
--cuda "cuda:0" \
--recompute \
--return_attention True \
--ckpt_path result/nougat/20231207_183828/epoch=201-step=56358.ckpt \
