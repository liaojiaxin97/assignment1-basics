
import torch
import torch.nn as nn
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps = None,device = None,dtype = None):
        super().__init__()
        factory_kwargs = {"device":device,"dtype":dtype}
        

        self.weight = nn.Parameter(torch.ones(d_model,**factory_kwargs))

        if eps is None:
            self.eps = 1e-5
        else: 
            self.eps = eps
        
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        #x:(batch_size,sequence_length,d_model)
        
        in_dtype = x.dtype
        
        x_float = x.to(torch.float32)
        
        #计算均方根 
        #rms  = sqrt(mean(x^2) + eps)
        #dim = -1 表示在隐藏层计算，keepdim =True 方便后续除法自动广播
        
        ms =x_float.pow(2).mean(dim = -1,keepdim = True)
        rms = torch.sqrt(ms+self.eps)
        
        #归一化并乘以可学习增益参数
        
        result = (x_float / rms) * self.weight
        
        return result.to(in_dtype)
        