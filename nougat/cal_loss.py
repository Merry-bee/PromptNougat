from nougat.position_decoder import iou_loss,diou_loss
from torch.nn import CrossEntropyLoss

def cal_loss(logits,labels,prompt_pred,prompt_true):
    loss_fct = CrossEntropyLoss()
    loss_token = loss_fct(logits, labels)   # logits[bs*label_len,50000],labels[bs*label_len]
    loss_position = diou_loss(pred=prompt_pred,target=prompt_true)
    
    return loss_token,loss_position