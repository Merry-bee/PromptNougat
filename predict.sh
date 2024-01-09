python -m pdb predict.py data/smiles_pic_cjd/demo/CHEMBL9_0.pdf \
--out output/greedy_search \
--checkpoint result/smiles/20240105_190348 \
--batchnum 8 \
--batchsize 1 \
--cuda "cuda:0" \
--recompute \
--return_attention True \
--ckpt_path result/smiles/20240105_190348/epoch=29-step=22980.ckpt \
