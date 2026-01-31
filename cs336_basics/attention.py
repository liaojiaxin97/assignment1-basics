import torch
import math

def softmax(x,dim):
    x_max = torch.max(x,dim=dim,keepdim=True).values
    #先减去最大值再取指数，保证指数计算稳定
    #tips:分子分母都乘了同一个常数会相互抵消，所以减去最大值只做数值稳定性处理，不改变结果
    exp_x = torch.exp(x - x_max)
    return exp_x/torch.sum(exp_x,dim = dim ,keepdim= True)

def scaled_dot_product_attention(
    Q:torch.Tensor,
    K:torch.Tensor,
    V:torch.Tensor,
    mask:torch.Tensor = None
) -> torch.Tensor:
    """
    Q:[...,n,d_K]
    k:[...,m,d_K]
    v:[...,m,d_v]
    mask:[n,m]
    """
    d_k = Q.size(-1)
    
    scores = torch.einsum("...nk,...mk -> ...nm", Q , K )/ math.sqrt(d_k)
    
    if mask is not None:
        #False 对应分数设为负无穷，再softmax概率为0
        scores = scores.masked_fill(mask == False,float("-inf"))
    
    probs = softmax(scores,dim = -1)
    
    output = torch.einsum("...nm,...mk -> ...nk",probs,V)
    
    return output