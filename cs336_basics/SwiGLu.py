import torch
import torch.nn as nn
from einops import rearrange
from tests.linear import Linear
def silu_fn(in_feature):
    
    return in_feature * torch.sigmoid(in_feature)

class SwiGLU(nn.Module):
    def __init__(self,d_model: int, d_ff:int,device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        #w1\w3 并行升维层
        self.w1 = Linear(self.d_model,self.d_ff,device = device,dtype = dtype)
        self.w3 = Linear(self.d_model,self.d_ff,device = device,dtype = dtype)
        #w3 降维层
        self.w2 = Linear(self.d_ff,self.d_model,device = device,dtype = dtype)
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        gate = silu_fn(self.w1(x))
        
        output = self.w3(x)
        
        return self.w2(gate * output)