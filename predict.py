"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
from pathlib import Path
import logging
import re
import argparse
import re
import pickle
from functools import partial
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from nougat import PromptNougatModel
from nougat.utils.dataset import LazyDataset
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import fitz
import numpy as np
from utils.functions import perplexity as perplexity_loss
import json
logging.basicConfig(level=logging.INFO)

if torch.cuda.is_available():
    BATCH_SIZE = int(
        torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1000 * 0.3
    )
else:
    # don't know what a good value is here. Would not recommend to run on CPU
    BATCH_SIZE = 5
    logging.warning("No GPU found. Conversion on CPU is very slow.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",  # default = 7
        "-b",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to use.",
    )
    parser.add_argument(
        "--batchnum",  # = len(dataloader) = sum(pagenum)/batchsize
        type=int,
        help="Num of batches to construct a dataloader, 防止dataloader太大爆掉",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed PDF, discarding previous predictions.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Add postprocessing step for markdown compatibility.",
    )
    parser.add_argument(
        "--cuda",
        default = "cuda:0"
    )
    parser.add_argument(
        "--return_attention",
        default = False
    )
    
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF(s) to process.")
    args = parser.parse_args()
    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint)
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if args.return_attention:   # return attention page by page
        args.batchsize = 1
        args.recompute = True
    if len(args.pdf) == 1 and not args.pdf[0].suffix == ".pdf":
        # input is a list file of pdfs
        try:
            args.pdf = [
                Path(l) for l in open(args.pdf[0]).read().split("\n") if len(l) > 0
            ]
        except:
            print('Invalid pdf path!')
    return args

def predict_files(datasets,args,model,pdf=None):
    
    dataloader = torch.utils.data.DataLoader(
        ConcatDataset(datasets),
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    predictions = []
    scores = []
    file_index = 0
    page_num = 0
    pdf_error = False
    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
        # there may be multiple pdf files in one batch: is_last_page:('','','aaa.pdf','','','bbb.pdf','')
        # one pdf file may also be divided into multiple batches: is_last_page:('','','ccc.pdf','','','','')
        # sample: [bs,3,896,672]
        model_output = model.inference(args,image_tensors=sample,return_attentions=args.return_attention,pdf=(pdf,i))
        
        # 存储scores和attentions
        # 注：存attention时batch_size=1；
        if 'attentions' in model_output.keys():  # return_attentions, score of one page
            score = {
                'logits': model_output['logits'],
                'decoder_attention': model_output["attentions"]["self_attentions"],
                'cross_attention': model_output["attentions"]["cross_attentions"]
            }
            scores.append(score)
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
        # for j in range(len(model_output)):
            # output = model_output[j]['predictions']
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages"
                    % (datasets[file_index].name, datasets[file_index].size)
                )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                pdf_error = True
                perplexity = None
               
            elif model_output["repeats"][j] is not None:
                # if model_output["repeats"][j] > 0:
                # If we end up here, it means the output is most likely not complete and was truncated.
                predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                predictions.append(output)
                predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                '''
                else:   # model_output["repeats"][j] = 0
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    # predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n")
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                    predictions.append(output)
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")'''
                
                pdf_error = True
                perplexity = None
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)
                logits = torch.stack(model_output['logits'],dim=1)[j].unsqueeze(0)  # [batch_size,seq_len,vocab_size]
                labels = model_output['sequences'][j][1:].unsqueeze(0)              # [batch_size,seq_len]
                perplexity = np.exp(perplexity_loss(logits,labels,pad_id=model.decoder.tokenizer.pad_token_id).cpu().to(torch.float64))
                
            if is_last_page[j]:
                # one pdf file compeleted, clear the predictions and pdf_error
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                if args.return_attention:
                    # 每个json一个pdf
                    if pdf_error:
                        score_path = args.out / Path('error_scores') / Path(is_last_page[j]).with_suffix(".pkl").name
                    else:
                        score_path = args.out / Path('correct_scores') / Path(is_last_page[j]).with_suffix(".pkl").name 
                    with open(score_path,'wb') as fo:
                        pickle.dump(scores,fo)  
                if args.out:   
                    if pdf_error:
                        out_path = args.out / Path('error') / Path(is_last_page[j]).with_suffix(".mmd").name
                    else:
                        out_path = args.out / Path('correct') / Path(is_last_page[j]).with_suffix(".mmd").name
                        
                        
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(out, encoding="utf-8")
                    
                    perplexity = perplexity if perplexity else np.inf
                    '''with open(f'{args.out}/1093_perplexity.jsonl','a')as fp:
                        json.dump({is_last_page[j]:float(perplexity)},fp)
                        fp.write('\n')'''
                else:
                    print(out, "\n\n")
                predictions = []
                scores = []
                page_num = 0
                file_index += 1
                pdf_error = False
            elif perplexity:
                pass
                '''with open(f'{args.out}/1093_perplexity.jsonl','a')as fp:
                    json.dump({page_num:float(perplexity)},fp)
                    fp.write('\n')'''
      

def main():
    args = get_args()
    model = PromptNougatModel.from_pretrained(args.checkpoint).to(torch.bfloat16)
    # 保存模型时使用了pl.LightningModule，因此参数多了‘model.’
    if args.ckpt_path is not None:
        model.load_state_dict({re.sub(r'^model.decoder','decoder',re.sub(r'^model.encoder','encoder',k)):v for k,v in torch.load(args.ckpt_path)['state_dict'].items()})
  
    if torch.cuda.is_available():
        model.to(args.cuda)
    model.eval()
    datasets = []
    pagenum=0
    for pdf in args.pdf:
        if not pdf.exists():
            print(f'File {pdf} not exists.')
            continue
        if args.out:
            out_path_correct = args.out  / Path('correct') / pdf.with_suffix(".mmd").name
            out_path_error = args.out  / Path('error') / pdf.with_suffix(".mmd").name
            if not args.recompute and (out_path_correct.exists() or out_path_error.exists()):
                logging.info(
                    f"Skipping {pdf.name}, already computed. Run with --recompute to convert again."
                )
                continue
        try:
            dataset = LazyDataset(
                pdf, partial(model.encoder.prepare_input, random_padding=False)
            )
        except fitz.fitz.FileDataError:
            logging.info(f"Could not load file {str(pdf)}.")
            continue
        datasets.append(dataset)

        # try:
            # 每个pdf inference一次
        predict_files(datasets,args,model,pdf=str(pdf))    
        # except Exception as e:
            # print(e)
        datasets=[]

        # 每batchnum*batchsize页inference一次
        '''
        pagenum += len(dataset)
        if pagenum/args.batchsize >= args.batchnum:
            predict_files(datasets,args,model)
            datasets=[]
            pagenum=0'''
        
    # 剩余不被batch_size整除的部分
    '''if len(datasets)>0:
        predict_files(datasets,args,model)'''
    
if __name__ == "__main__":
    # data/quantum/10.1088/0305-4470%2F37%2F38%2FL01.pdf
    # data/quantum/10.1103/PhysRevApplied.15.L021003.pdf
    main()
