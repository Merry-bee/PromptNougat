import torch
from torch import Tensor
import numpy as np
import torch.nn.functional as F


def perplexity(outputs: Tensor, targets: Tensor, pad_id=None):
    """
    计算语言模型困惑度
    :param outputs: [batch_size,seq_len,vocab_size]
    :param targets: [batch_size,seq_len]
    :param pad_id:  model.decoder.tokenizer.pad_id
    :return: 困惑度数值
    """
    ce = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1),
                         ignore_index=pad_id if pad_id else None)

    return torch.exp(ce)
