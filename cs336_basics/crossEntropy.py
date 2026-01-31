import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    
    """
    logits: (batch,seq,vocabsize)
    targets: (batch,seq)
    
    """
    
    #1.计算每组logits的最大值，保持数值稳定
    
    m = torch.max(logits, dim = -1,keepdim= True).values
    
    
    #2. 提取目标位置对应的原始分值
    target_logits = torch.gather(logits, dim = -1, index= targets.unsqueeze(-1)).squeeze(-1)
    
    #3.计算log-sum-exp
    
    shifted_logits = logits - m
    
    #公式：M + log(sum(exp(o - M))
    #m (B,S,1)  求和
    #torch.sum 默认行为是 keepdim=False。也就是求和的那一维会直接消失
    log_sum_exp = m.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits),dim = -1))
    
    #shape [B,s]
    loss = log_sum_exp - target_logits
    
    #求全批次平均 (loss1+loss2+……+lossn) / n
    
    return torch.mean(loss)