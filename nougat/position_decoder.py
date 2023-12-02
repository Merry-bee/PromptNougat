import torch
from torch import nn
import torch.nn.functional as F

class PositionDecoder(nn.Module):
    def __init__(self,decoder_attention_heads,decoder_layers,input_dim=588, hidden_dim=256, output_dim=4, num_layers=3,image_size=[896,672]):
        super().__init__()
        self.head_linear = nn.Linear(decoder_attention_heads*decoder_layers,1)   # 对16个head*4个layer加权
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])) # [(input_dim,hidden_dim),(hidden_dim,hidden_dim),(hidden_dim,output_dim)]
        self.image_size = image_size
        
    
    def forward(self, heatmap):
        '''
        args:
            heatmap:[bs,16,len(input_ids),588]
        '''
        num_layer,bs,num_heads,input_len,encoder_len = heatmap.shape  # [4,bs,16,len,588]
        heatmap = heatmap.permute(1,3,4,0,2).reshape(bs,input_len,encoder_len,-1)  # [bs,len,588,4,16] -> [bs,len,588,64]
        x = self.head_linear(heatmap).squeeze(-1)   # [bs,len,588,64]->[bs,len,588,1]->[bs,len,588]
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = x.sigmoid() # [bs,len,4] 标准化的坐标
        x1 = x[:,:,0]*self.image_size[1]   # [bs,len]
        y1 = x[:,:,1]*self.image_size[0]
        x2 = x[:,:,2]*self.image_size[1]
        y2 = x[:,:,3]*self.image_size[0]
        return x1,y1,x2,y2
        
        
def iou_loss(pred,target,epsilon=1e-5):
    '''
    args: 
    pred/target: [bs,length,2,2]
    '''
    
    pred = pred.reshape(-1,2,2) # [bs*len,2,2]
    target = target.reshape(-1,2,2) 
    
    inter_x1 = torch.max(pred[:,0,0],target[:,0,0])
    inter_y1 = torch.max(pred[:,0,1],target[:,0,1])
    inter_x2 = torch.min(pred[:,1,0],target[:,1,0])
    inter_y2 = torch.min(pred[:,1,1],target[:,1,1])
    # 确保交集面积不小于0
    inter_area = torch.clamp(inter_x2-inter_x1,min=0)*torch.clamp(inter_y2-inter_y2,min=0)
    pred_area = (pred[:,1,0]-pred[:,0,0])*(pred[:,1,1]-pred[:,0,1])
    target_area = (target[:,1,0]-target[:,0,0])*(target[:,1,1]-target[:,0,1])
    union_area = pred_area + target_area - inter_area
    iou = inter_area/(union_area+epsilon)
    iou_loss = (1-iou).mean()

    return iou_loss
    
def diou_loss(pred,target,epsilon=1e-5):
    '''
    args: 
    pred/target: [bs,length,2,2]
    '''
    
    pred = pred.reshape(-1,2,2) # [bs*len,2,2]
    target = target.reshape(-1,2,2) 
    
    inter_x1 = torch.max(pred[:,0,0],target[:,0,0])
    inter_y1 = torch.max(pred[:,0,1],target[:,0,1])
    inter_x2 = torch.min(pred[:,1,0],target[:,1,0])
    inter_y2 = torch.min(pred[:,1,1],target[:,1,1])
    # 确保交集面积不小于0
    inter_area = torch.clamp(inter_x2-inter_x1,min=0)*torch.clamp(inter_y2-inter_y2,min=0)
    pred_area = (pred[:,1,0]-pred[:,0,0])*(pred[:,1,1]-pred[:,0,1])
    target_area = (target[:,1,0]-target[:,0,0])*(target[:,1,1]-target[:,0,1])
    union_area = pred_area + target_area - inter_area
    iou = inter_area/(union_area+epsilon)
    
    pred_center_x = (pred[:,1,0]+pred[:,0,0])/2
    pred_center_y = (pred[:,1,1]+pred[:,0,1])/2
    target_center_x = (target[:,1,0]+target[:,0,0])/2
    target_center_y = (target[:,1,1]+target[:,0,1])/2
    d2 = torch.square(pred_center_x-target_center_x)+torch.square(pred_center_y-target_center_y)
    
    out_x1 = torch.min(pred[:,0,0],target[:,0,0])
    out_y1 = torch.min(pred[:,0,1],target[:,0,1])
    out_x2 = torch.max(pred[:,1,0],target[:,1,0])
    out_y2 = torch.max(pred[:,1,1],target[:,1,1])
    c2 = torch.square(out_x2-out_x1)+torch.square(out_y2-out_y1)
    
    diou_loss = (1-iou+d2/c2).mean()


    return diou_loss

         
        




   
        
    


    