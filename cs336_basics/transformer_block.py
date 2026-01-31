import torch
import torch.nn as nn
import einops
from einops import rearrange
from tests.attention import scaled_dot_product_attention
from tests.linear import Linear
from tests.Rope import RoPE
from tests.CausalSelfAttention import CausalSelfAttention
from tests.RMSNorm import RMSNorm
from tests.SwiGLu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, 
                 theta: float, user_rms_norm, norm_mode, ffn_type, 
                 device = None, dtype = None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
    
        #预处理
        self.RMSNorm1 = RMSNorm(d_model = self.d_model, device=self.device, dtype= dtype)
        self.RMSNorm2 = RMSNorm(d_model = self.d_model, device=self.device, dtype= dtype)
        #Casual MutiHaed Attention
        self.MultiheadAttention = CausalSelfAttention(d_model = self.d_model, num_head = self.num_heads,max_seq_len = max_seq_len, theta=theta)
        #前馈神经网络
        self.FFN = SwiGLU(d_model = self.d_model, d_ff = self.d_ff, device=self.device, dtype= dtype)

    def forward(self,x:torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        b,s,d = x.shape
        
        x_residual = x
        x = self.MultiheadAttention(self.RMSNorm1(x),token_positions=token_positions)
        output1 = x + x_residual
        
        x_residual = output1
        output2 = self.FFN(self.RMSNorm2(output1))
        attn_output = output2 + x_residual
        
        return attn_output
        # x = x + self.MultiheadAttention(self.RMSNorm1(x),token_positions)
        # x = x + self.FFN(self.RMSNorm2(x))
        # return x