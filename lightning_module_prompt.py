"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import math
import random
from pathlib import Path
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
from pytorch_lightning.utilities import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from nougat import PromptNougatConfig, PromptNougatModel
from nougat.metrics import get_metrics
from nougat.cal_loss import cal_loss


class PromptModelPLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.validation_step_outputs = []
        self.config = config
        if self.config.get("model_path", False):
            self.model = PromptNougatModel.from_pretrained(
                self.config.model_path,
                input_size=self.config.input_size,
                max_length=self.config.max_length,
                align_long_axis=self.config.align_long_axis,
                window_size=self.config.window_size,
                encoder_layer=self.config.encoder_layer,
                decoder_layer=self.config.decoder_layer,
                patch_size=self.config.patch_size,
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                hidden_dimension=self.config.hidden_dimension,
                ignore_mismatched_sizes=True,
            )
            # old_params = torch.load(self.config.model_path+'/pytorch_model.bin').keys()
            # for n,p in self.model.named_parameters():
            #     if n not in old_params or re.match(r'decoder\.model\.model\.decoder\.layers\.[0-3]\.encoder_attn\.',n) is not None:
            #         p.requires_grad = True # 训练新加入参数; decoder.cross_attn加入训练
            #     else:
            #         p.requires_grad = False  # 冻结已有参数
            if self.config.ckpt_path is not None:
                self.model.load_state_dict({re.sub(r'^model.decoder','decoder',re.sub(r'^model.encoder','encoder',k)):v for k,v in torch.load(self.config.ckpt_path)['state_dict'].items()})
    
        else:
            self.model = PromptNougatModel(
                config=PromptNougatConfig(
                    input_size=self.config.input_size,
                    max_length=self.config.max_length,
                    align_long_axis=self.config.align_long_axis,
                    window_size=self.config.window_size,
                    encoder_layer=self.config.encoder_layer,
                    decoder_layer=self.config.decoder_layer,
                    tokenizer_file=self.config.tokenizer,
                    patch_size=self.config.patch_size,
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    hidden_dimension=self.config.hidden_dimension,
                )
            )

    def training_step(self, batch, batch_idx):
        image_tensors, pre_input_ids, label_ids, attention_masks,prompts,keep_row_label = list(), list(), list(), list(), list(), list()
        if batch is None:
            return
        for batch_data in batch:    # batch: input_tensor, pre_ids, attention_mask, label_id, prompt
            if batch_data is None or batch_data[0] is None:
                continue
            image_tensors.append(batch_data[0])     # image
            pre_input_ids.append(batch_data[1])     # pre_ids
            attention_masks.append(batch_data[2])   # attention_mask
            label_ids.append(batch_data[3])         # label_id
            prompts.append(batch_data[4])           # prompt
            keep_row_label.append(batch_data[5])   # keep_row_label
        image_tensors = torch.cat(image_tensors)
        pre_input_ids = torch.cat(pre_input_ids)
        attention_masks = torch.cat(attention_masks)
        label_ids = torch.cat(label_ids)
        prompts = torch.cat(prompts)
        keep_row_label = torch.cat(keep_row_label)
        loss,loss_token,loss_position,iou = self.model(image_tensors, pre_input_ids, attention_masks, label_ids, prompt_in=prompts[:,:-1,:,:],prompt_true=prompts[:,1:,:,:],keep_row_label=keep_row_label)[0] #CrossEntropyLoss()+IoU_loss()
        if loss is not None:
            self.log_dict({"train/loss": loss}, sync_dist=False)
            self.log_dict({"train/loss_token": loss_token}, sync_dist=False)
            self.log_dict({"train/loss_position": loss_position}, sync_dist=False)
            self.log_dict({"train/iou": iou}, sync_dist=False)
            return loss
        else:
            self.log_dict({"train/loss_token": loss_token}, sync_dist=False)
            return loss_token

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if batch is None:
            return
        image_tensors,input_ids, attention_masks, labels, prompts,keep_row_label = batch
        prompt_in = prompts[:,:-1,:,:]
        prompt_true=prompts[:,1:,:,:]
        if image_tensors is None:
            return
     
        # validation data with prompt, loss=one token
        # 处理方式同model.py/PromptNougatModel/forward()
        output = self.model.inference(
            image_tensors=image_tensors,    # shape=[bs,3,588,1024]
            input_ids = input_ids,
            attention_mask=attention_masks,
            return_attentions=True,
            prompt=prompt_in,
            keep_row_label=keep_row_label,
            validation=True,
        )
        logits = output["logits"][0]  # pred id: [bs,len,50000]    
        
        if prompt_in.shape[1] != logits.shape[1]:  # 非全文prompt 
            # validation with prompt，取对应位置的logits和labels
            bs =  logits.shape[0]        
            prompt_label_length = labels.shape[1]                           
            pred = torch.zeros(bs,prompt_label_length,logits.shape[-1])
            pred[:,:,self.model.decoder.tokenizer.pad_token_id] = 100
            for b in range(bs):
                start_idx = max(torch.where(attention_masks[b,:]==1)[0]) # target token start, 但input_ids拼接了labels
                if self.model.decoder.tokenizer.pad_token_id in labels[b]:          
                    label_len = min(torch.where(labels[b]==self.model.decoder.tokenizer.pad_token_id)[0])-2     # 除去<s>和</s>的长度
                else:
                    label_len = prompt_label_length-2
                pred[b,:label_len,:] = logits[b,start_idx-label_len:start_idx,:]                                   # [bs,prompt_label_len,50000]
                labels[b] = torch.cat((labels[b,1:label_len+1],torch.full([prompt_label_length-label_len],self.model.decoder.tokenizer.pad_token_id).to(labels.device)))      # [bs,prompt_label_len]，将[<s>,x,x,x,</s>,<pad>]改为[x,x,x,<pad>,<pad><,pad>]
            logits = pred.to(logits.device) # [bs,prompt_label_len,50000]

        gts = self.model.decoder.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        ) # label token  
        preds = self.model.decoder.tokenizer.batch_decode(
            torch.argmax(logits,dim=-1), skip_special_tokens=True
        )    # pred token 
                                        
        metrics = get_metrics(gts, preds, pool=False)
        scores = {
            "val/" + key: sum(values) / len(values) for key, values in metrics.items()
        }
        loss, loss_token,loss_position,iou = cal_loss(logits=logits.view(-1,self.model.decoder.tokenizer.vocab_size),labels=labels.view(-1),prompt_pred=output['prompt_pred'],prompt_true=prompt_true,p_keep_row = output['p_keep_row'],keep_row_label=keep_row_label.view(-1))
        scores["val/loss_token"] = loss_token
        scores["val/loss_position"] = loss_position
        scores["val/iou"] = iou
        scores["val/loss"] = loss
        
        self.validation_step_outputs.append(scores)
        self.log('val_loss',loss)
        return scores

    def on_validation_epoch_end(self):
        if (
            self.validation_step_outputs is not None
            and len(self.validation_step_outputs) >= 1
        ):
            self.log_dict(self.validation_step_outputs[0], sync_dist=False)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        max_iter = None

        if int(self.config.get("max_epochs", -1)) > 0:
            assert (
                len(self.config.train_batch_sizes) == 1
            ), "Set max_epochs only if the number of datasets is 1"
            steps = self.config.num_training_samples_per_epoch
            max_iter = (self.config.max_epochs * steps) / max(
                1,
                (
                    self.config.train_batch_sizes[0]
                    * torch.cuda.device_count()
                    * self.config.get("num_nodes", 1)
                ),
            )

        if int(self.config.get("max_steps", -1)) > 0:
            max_iter = (
                min(self.config.max_steps, max_iter)
                if max_iter is not None
                else self.config.max_steps
            )

        assert max_iter is not None
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,self.parameters()), lr=self.config.lr)
        scheduler = {
            # "scheduler": self.exponential_scheduler(
            #     optimizer,
            #     self.config.warmup_steps,
            #     self.config.lr,
            #     self.config.get("min_lr", 5e-5),
            #     self.config.get("gamma", 0.9996),
            # ),
            "scheduler": self.cosine_scheduler(
                optimizer,
                self.config.warmup_steps*10, 
                self.config.warmup_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": self.config.get("lr_step", 1),
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)   # 余弦周期的长度

            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def exponential_scheduler(optimizer, warmup_steps, lr, min_lr=5e-5, gamma=0.9999):
        def lr_lambda(x):
            if x > warmup_steps or warmup_steps <= 0:
                if lr * gamma ** (x - warmup_steps) > min_lr:
                    return gamma ** (x - warmup_steps)
                else:
                    return min_lr / lr
            else:
                return x / warmup_steps

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = (
            Path(self.config.result_path)
            / self.config.exp_name
            / self.config.exp_version
        )
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)


class PromptDataPLModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_batch_sizes = self.config.train_batch_sizes
        self.val_batch_sizes = self.config.val_batch_sizes
        self.train_datasets = []
        self.val_datasets = []
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

    def train_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.train_datasets),
                batch_size=self.train_batch_sizes[0],
                num_workers=self.config.num_workers,
                pin_memory=True,
                worker_init_fn=self.seed_worker,
                generator=self.g,
                shuffle=True,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    def val_dataloader(self):
        loaders = [
            DataLoader(
                torch.utils.data.ConcatDataset(self.val_datasets),
                batch_size=self.val_batch_sizes[0],
                pin_memory=True,
                shuffle=False,
                collate_fn=self.ignore_none_collate,
            )
        ]
        return loaders

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def ignore_none_collate(batch):
        if batch is None:
            return
        try:
            batch = [x for x in batch if x is not None and x[0] is not None]
            if len(batch) == 0:
                return
            return torch.utils.data.dataloader.default_collate(batch)
        except AttributeError:
            pass
