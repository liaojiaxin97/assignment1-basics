

import torch
import torch.nn as nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float,d_k:int, max_seq_len:int,device= None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        #此处1.0是，而公式是i,在sinusoid中有外乘积，保证分子随位置变换而变化
        freqs = 1.0 / (self.theta ** (torch.arange(0,d_k,2,device=self.device).float()/self.d_k))
        
        position = torch.arange(self.max_seq_len)
        
        sinusoid = torch.outer(position,freqs)

        self.register_buffer("cos_cache",sinusoid.cos(),persistent = False)
        self.register_buffer("sin_cache",sinusoid.sin(),persistent = False)
        
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:
        #x(b,s,d)
        #cos s,d
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        #cos 1,s,d
        #2维对齐三维左边补1
        #2维对齐4维,需要在第三维补1,然后第四维会自动补(batch)
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1) # 第三维补齐
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        #假设 x 的形状是 (b, h, s, d_k)，那么 x1 和 x2 的形状都是 (b, h, s, d_k / 2)。
        x1,x2 = x[...,::2],x[...,1::2]
        #经过旋转计算后，output1 和 output2 的形状也都是 (b, h, s, d_k / 2)
        output1 = x1 * cos - x2 * sin
        output2 = x1 * sin + x2* cos
        #将 output1 和 output2 堆叠起来，生成一个新张量 out。这个新张量的形状会变为 (b, h, s, d_k / 2, 2)。
        out = torch.stack([output1,output2],dim = -1)
        #从倒数第二个维度开始展平，一直到最后一个维度 -->  (b, h, s, d_k)
        out = out.flatten(-2)
        
        return out 