"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
import json
import os
import logging
from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import torch
from tqdm import tqdm

from nougat import PromptNougatModel
from nougat.metrics import compute_metrics
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import NougatDataset
from lightning_module_prompt import PromptDataPLModule, PromptModelPLModule
from nougat.cal_loss import cal_loss
from nougat.visualization import visual_box

def show_gt(model,label_ids,prompt_true):
    token_lst = model.decoder.tokenizer.batch_decode(
            [[l] for l in label_ids], skip_special_tokens=True
        )
    for token,coord in zip(token_lst,prompt_true):
        coord = [[round(coord[0][0],2),round(coord[0][1],2)],[round(coord[1][0],2),round(coord[1][1],2)]]
        print(f'This token:"{token}",next coord:"{coord}"')    

def test(args):
    pretrained_model = PromptNougatModel.from_pretrained(args.model_path).to(torch.bfloat16)
    if args.ckpt_path is not None:
        pretrained_model.load_state_dict({re.sub(r'^model.decoder','decoder',re.sub(r'^model.encoder','encoder',k)):v for k,v in torch.load(args.ckpt_path)['state_dict'].items()})
    if torch.cuda.is_available():
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    else:
        logging.warning("Results can not be saved. Please provide a -o/--save_path")
    predictions = []
    ground_truths = []
    metrics = defaultdict(list)
    dataset = NougatDataset(
        dataset_path=args.dataset,
        nougat_model=pretrained_model,
        max_length=pretrained_model.config.max_length,
        split=args.split,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=6,
        pin_memory=True,
        shuffle=args.shuffle,
        collate_fn=PromptDataPLModule.ignore_none_collate,
    )
    if args.visualize:
        with open(args.dataset,'r') as fi:
            data_jsonl=fi.readlines()
    # 同validation_step()
    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        sample = [n.to(pretrained_model.device) for n in sample]
        image_tensors, input_ids,attention_masks,label_ids,prompts = sample
        if image_tensors is None:
            return
        # if len(predictions) >= args.num_samples:
        #     break
        outputs = pretrained_model.inference(
            image_tensors=image_tensors,    
            input_ids = input_ids,
            attention_mask=attention_masks,
            return_attentions=True,
            prompt=prompts[:,:-1,:,:].bfloat16(),
            validation=True,
        )
        
        
        # label_ids[0] vs. pred_label_ids
        logits = outputs["logits"][0].view(-1,pretrained_model.decoder.tokenizer.vocab_size)  # pred id: [bs,len,50000] 
        pred_label_ids = logits.max(dim=1).indices   # [seq_len]
        
        # sequences[0] vs. ground_truth[0]
        sequences = pretrained_model.decoder.tokenizer.batch_decode(pred_label_ids.unsqueeze(0), skip_special_tokens=True)
        predictions.extend(sequences)
        ground_truth = pretrained_model.decoder.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        ground_truths.extend(ground_truth)
        
        loss, loss_token,loss_position,iou = cal_loss(logits=logits,labels=label_ids.view(-1),prompt_pred=outputs['prompt_pred'],prompt_true=prompts[:,1:,:,:])
        if args.visualize:
            png_path = json.loads(data_jsonl[idx])['image']
            visual_box(png_path=Path(args.dataset).parent/png_path,prompt_pred=outputs['prompt_pred'],prompt_true=prompts[:,1:,:,:],save_path=f'data/tmp/visual_png/{Path(png_path).name}',image_size = [672,896])
        print(f'loss_token:{loss_token},loss_position:{loss_position},loss:{loss}')
        # visualization
        start,end = 0,11
        '''show_gt(pretrained_model,label_ids[0].tolist()[start:end],prompt_true = prompts[0,1:,:,:].tolist()[start:end])
        
        with Pool(args.batch_size) as p:
            _metrics = p.starmap(compute_metrics, iterable=zip(outputs, ground_truth))
            for m in _metrics:
                for key, value in m.items():
                    metrics[key].append(value)

            print({key: sum(values) / len(values) for key, values in metrics.items()})

    scores = {}
    for metric, vals in metrics.items():
        scores[f"{metric}_accuracies"] = vals
        scores[f"{metric}_accuracy"] = np.mean(vals)
    try:
        print(
            f"Total number of samples: {len(vals)}, Edit Distance (ED) based accuracy score: {scores['edit_dist_accuracy']}, BLEU score: {scores['bleu_accuracy']}, METEOR score: {scores['meteor_accuracy']}"
        )
    except:
        pass
    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        with open(args.save_path, "w") as f:
            json.dump(scores, f)

    return predictions'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--save_path", "-o", type=str, default=None, help="json file to save results to"
    )
    parser.add_argument("--num_samples", "-N", type=int, default=-1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--batch_size", "-b", type=int, default=10)
    parser.add_argument("--visualize",  type=bool, default=False)
    args, left_argv = parser.parse_known_args()
    args.model_path = get_checkpoint(args.model_path)

    predictions = test(args)
