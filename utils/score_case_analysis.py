import pickle
import torch
import os



# plot attentions
def plot_layer_attn(score_file,page,start_idx=0,length=20,checkpoint='checkpoints/0.1.0-small'):
    with open(score_file,'rb') as fi:
        scores = pickle.load(fi)   
    logits = torch.stack(scores[page]['logits'], 1).cpu().max(-1)   # [batch_size,seq_len]
    end_idx = min(len(scores[page]['cross_attention']),start_idx+length)
    for token_idx in range(start_idx,end_idx):
        # cross_attention: [num_layers,num_heads,588] -> [num_layers,num_heads,28,21]
        cross_attention = torch.stack(scores[page]['cross_attention'][token_idx]).permute(1, 3, 0, 2, 4).squeeze(0).squeeze(0).view(4, 16, 28, 21)
        
        import matplotlib.pyplot as plt
        for layer in range(cross_attention.shape[0]):
            for head in range(cross_attention.shape[1]):
                # frame:[28,21]
                frame = cross_attention[layer,head,:,:].cpu().to(torch.float)
                # plt.clf()
                # plt.imshow(frame,cmap='viridis')
                # plt.savefig(f'output/tmp/token{token_idx}_head{head}_layer{layer}.png')


def plot_fused_attn(score_file,page):
    with open(score_file,'rb') as fi:
        scores = pickle.load(fi)   

    fusedattention = []
    for token_idx in range(len(scores[page]['cross_attention'])):
        # cross_attention: [num_layers,num_heads,588] -> [num_layers,num_heads,28,21]
        cross_attention = torch.stack(scores[page]['cross_attention'][token_idx]).permute(1, 3, 0, 2, 4).squeeze(0).squeeze(0).view(4, 16, 28, 21)
        # 先对head取最大值，再对layer取最大值
        fusedattention.append(cross_attention.max(1)[0].max(0)[0].cpu().to(torch.float))


    import matplotlib.pyplot as plt

    file_name = score_file.split('/')[-1][:-4]
    error = score_file.split('/')[2].split('_')[0]=='error'
    if error:
        out_dir = f"{score_file.split('error')[0]+'error_attns'}/{file_name}_{page}"
    else:
        out_dir = f"{score_file.split('correct')[0]+'correct_attns'}/{file_name}_{page}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, frame in enumerate(fusedattention):
        plt.clf()
        plt.imshow(frame, cmap='viridis')
        plt.title(f'Frame {i+1}/{len(fusedattention)}')
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(f'{out_dir}/{i}.png')

def decode(score_file,page,checkpoint='checkpoints/0.1.0-small'):
    with open(score_file,'rb') as fi:
        scores = pickle.load(fi)   
    logits = torch.stack(scores[page]['logits'], 1).cpu().max(-1)   # [batch_size,seq_len]
    from nougat import NougatModel
    model = NougatModel.from_pretrained(checkpoint).to(torch.bfloat16)
    output = ' '.join([model.decoder.tokenizer.decode(x) for x in logits.indices])
    print('\n\n'+output)
    print(f'len(output):{len(output)}')

def gt_prob(score_file,page,gt,start_idx,checkpoint='checkpoints/0.1.0-small'):
    from nougat import NougatModel
    model = NougatModel.from_pretrained(checkpoint).to(torch.bfloat16)
    true_idxs = model.decoder.tokenizer(gt)['input_ids'][1:-1]  # [0,idxs,2],去掉CLS和EOS

    with open(score_file,'rb') as fi:
        scores = pickle.load(fi)   
    logits_prob = torch.stack(scores[page]['logits'], 1).cpu().squeeze(0)  # [seq_len,50000]
    for i,words_score in enumerate(logits_prob[start_idx:start_idx+len(gt)+1]): # [length,50000]
        pred_score = words_score[true_idxs[i]]  # 50000, 153:5.1562
        max_scores = words_score.topk(3)
        print(pred_score in max_scores) # 判断ground_truth的预测概率，看看什么样的decoding strategy可以解出gt
        
if __name__ == '__main__':
    # load scores
    score_file='output/greedy_search/correct_scores/0009-2614%2881%2980004-8.pkl'
    page = 1
    '''
    with open(score_file,'rb') as fi:
        scores = pickle.load(fi)    # scores is a list
    for score in scores:        # score is a dict
        score['logits']                # (seq_len,[batch_size,50000])
        score['decoder_attention']     # (seq_len,num_layers,[batch_size,num_heads,1,cur_len])
        score['cross_attention']       # (seq_len,num_layers,[batch_size,num_heads,1,588])
    '''
    plot_fused_attn(score_file,page)
    # plot_layer_attn(score_file,page,start_idx=150,length=10)
    # decode(score_file,page,checkpoint='checkpoints/0.1.0-small')
    # gt_prob(score_file,page,gt='Table',start_idx=648,checkpoint='checkpoints/0.1.0-small')
