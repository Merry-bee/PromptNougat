from nougat.position_decoder import diou_loss
from torch.nn import CrossEntropyLoss
import torch

def cal_loss(logits,labels,prompt_pred,prompt_true,logits_keep_row,keep_row_label):
    # loss_token
    loss_fct = CrossEntropyLoss()
    loss_token = loss_fct(logits, labels)   # logits[bs*label_len,50000],labels[bs*label_len]
    
    # loss_position
    # prompt_true:[bs,seq_len,2,2]->[bs*seq_len,4], diff: [0,0,0,0](pad)或[-1,-1,-1,-1](mask)返回0，其他返回非0，torch.where: 取非0的index
    valid_mask = torch.unique(torch.where(torch.diff(prompt_true.reshape(-1,4)))[0])  
    if len(valid_mask)>2:   # 存在除</s></work>外的prompt输入，loss=avg1(loss_token)+avg2(loss_position)
        prompt_pred = prompt_pred.reshape(-1,2,2)[valid_mask]
        prompt_true = prompt_true.reshape(-1,2,2)[valid_mask]
        keep_row_label = keep_row_label[valid_mask]
        logits_keep_row = logits_keep_row[valid_mask]
        loss_position,focal_loss,diou_loss1,diou_loss2,iou = diou_loss(pred=prompt_pred,target=prompt_true,logits_keep_row=logits_keep_row,keep_row_label=keep_row_label)  
        loss = loss_token + loss_position
    else:   # 整个bs没有任何prompt输入
        loss,loss_position,focal_loss,diou_loss1,diou_loss2,iou = None,None,None,None,None,None
    
    
    return loss,loss_token,loss_position,focal_loss,diou_loss1,diou_loss2,iou