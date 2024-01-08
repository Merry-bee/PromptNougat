import torch
from torch import nn
import torch.nn.functional as F

class PositionDecoder(nn.Module):
    def __init__(self,decoder_attention_heads,decoder_layers,input_dim=588, hidden_dim=256, output_dim=4, num_layers=3,image_size=[896,672]):
        super().__init__()
        self.head_linear = nn.Linear(decoder_attention_heads*decoder_layers,1)   # 对16个head*4个layer加权
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.keep_row_linear  = nn.Linear(input_dim,1)   # 是否换行
        self.decoder1 = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])) # [(input_dim,hidden_dim),(hidden_dim,hidden_dim),(hidden_dim,output_dim)]
        self.decoder2 = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])) # [(input_dim,hidden_dim),(hidden_dim,hidden_dim),(hidden_dim,output_dim)]
        self.layernorms = nn.ModuleList(nn.LayerNorm(n) for n in [input_dim] + h) 
        self.image_size = image_size
        
    
    def forward(self, heatmap,attention_valid_mask,keep_row_thres=0.5):
        '''
        args:
            heatmap:[bs,16,len(input_ids),588]
        '''
        num_layer,bs,num_heads,input_len,encoder_len = heatmap.shape  # [4,bs,16,len,588]
        heatmap = heatmap.permute(1,3,4,0,2).reshape(bs,input_len,encoder_len,-1)  # [bs,len,588,4,16] -> [bs,len,588,64]
        x = self.head_linear(heatmap).squeeze(-1)   # [bs,len,588,64]->[bs,len,588,1]->[bs,len,588]
        x = x.reshape(bs*input_len,-1)   # [bs*len,588]
        
        p_keep_row = self.keep_row_linear(x).sigmoid().view(-1)  # [bs,len,588]->[bs,len,1]->[bs*seq_len]
        x1_indices = torch.where(p_keep_row > keep_row_thres)[0]    # 保留当前行
        x1 = x[x1_indices]     
        x2_indices = torch.where(p_keep_row <= keep_row_thres)[0]  # 换行
        x2 = x[x2_indices]    
        for i in range(len(self.decoder1)):
            x1 = self.layernorms[i](x1)
            x1 = F.gelu(self.decoder1[i](x1)) if i < self.num_layers - 1 else self.decoder1[i](x1)                     
        for i in range(len(self.decoder2)):
            x2 = self.layernorms[i](x2)
            x2 = F.gelu(self.decoder2[i](x2)) if i < self.num_layers - 1 else self.decoder2[i](x2)
        out_tensor = torch.zeros(bs*input_len,4,dtype=x.dtype).to(x.device)
        out_tensor[x1_indices] = x1
        out_tensor[x2_indices] = x2
        out_tensor = out_tensor.reshape(bs,input_len,-1)  # # [bs*len,588]-> [bs,len,588]
        
        out_tensor = out_tensor.sigmoid() # [bs,len,4] 标准化的坐标
        # attention_valid_mask=0 -> pred=[[0,0],[0,0]](pad坐标)
        if attention_valid_mask.shape[1]==4095:   # train/validation with whole sentence
            attention_valid_mask = torch.cat((attention_valid_mask[:,1:],torch.zeros([bs,1]).to(x.device)),dim=1) # attention_mask对应input_ids，对应prompt_true需要向后移动一位
        attention_valid_mask = attention_valid_mask.unsqueeze(-1).expand(-1,-1,4) #[bs,seq_len,4]
        out_tensor = torch.mul(out_tensor,attention_valid_mask)   
        x1 = out_tensor[:,:,0]   # [bs,len]
        y1 = out_tensor[:,:,1]
        w = out_tensor[:,:,2]
        h = out_tensor[:,:,3]
      
        x2 = x1+w
        y2 = y1+h
        return [x1,y1,x2,y2],p_keep_row
        
        
def iou(pred,target,epsilon=1e-5):
    '''
    args: 
    pred/target: [bs,length,2,2]
    
    '''
    inter_x1 = torch.max(pred[:,0,0],target[:,0,0])
    inter_y1 = torch.max(pred[:,0,1],target[:,0,1])
    inter_x2 = torch.min(pred[:,1,0],target[:,1,0])
    inter_y2 = torch.min(pred[:,1,1],target[:,1,1])
    # 确保交集面积不小于0
    inter_area = torch.clamp(inter_x2-inter_x1,min=0)*torch.clamp(inter_y2-inter_y1,min=0)
    pred_area = (pred[:,1,0]-pred[:,0,0])*(pred[:,1,1]-pred[:,0,1])
    target_area = (target[:,1,0]-target[:,0,0])*(target[:,1,1]-target[:,0,1])
    union_area = pred_area + target_area - inter_area
    iou = (inter_area/(union_area+epsilon))
    

    return iou
    
def diou_loss(pred,target,p_keep_row,keep_row_label,epsilon=1e-5,gamma=2):
    '''
    args: 
    pred/target: [bs,length,2,2]
    
    '''
    pred = pred.reshape(-1,2,2) # [bs*len,2,2]
    target = target.reshape(-1,2,2) 
    
    iou_tensor = iou(pred,target,epsilon) # [bs*len,1]
    
    pred,target = pred,target
    pred_center_x = (pred[:,1,0]+pred[:,0,0])/2
    pred_center_y = (pred[:,1,1]+pred[:,0,1])/2
    target_center_x = (target[:,1,0]+target[:,0,0])/2
    target_center_y = (target[:,1,1]+target[:,0,1])/2
    d2 = (torch.square(pred_center_x-target_center_x)+torch.square(pred_center_y-target_center_y))
    out_x1 = torch.min(pred[:,0,0],target[:,0,0])
    out_y1 = torch.min(pred[:,0,1],target[:,0,1])
    out_x2 = torch.max(pred[:,1,0],target[:,1,0])
    out_y2 = torch.max(pred[:,1,1],target[:,1,1])
    c2 = (torch.square(out_x2-out_x1)+torch.square(out_y2-out_y1))
    diou_loss = 1-iou_tensor+d2
    
    pt = p_keep_row
    pt[~keep_row_label] = (1-p_keep_row)[~keep_row_label]
    focal_loss = (1-pt)**gamma * diou_loss

    return focal_loss.mean(),iou_tensor.mean()

         
        




   
        
    


    