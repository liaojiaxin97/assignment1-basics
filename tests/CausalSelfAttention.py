import torch
import torch.nn as nn
import einops
from einops import rearrange
from tests.attention import scaled_dot_product_attention
from tests.linear import Linear
from tests.Rope import RoPE
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model:int , num_head:int, max_seq_len = None, theta = None, device = None, dtype = None):
        #super().__init__() 是在任何模块属性（如 self.q_proj = Linear(...)）被赋值之前调用的第一条语句。
        super().__init__()
        #维度校验
        assert d_model% num_head == 0
        #初始参数
        self.d_model = d_model
        self.num_heads = num_head
        self.d_k = self.d_model/self.num_heads

        
        #Q/K/V投影，将输入映射到三个不同的特征空间，Linear
        self.q_proj = Linear(self.d_model,self.d_model,device = None,dtype = None)    
        self.k_proj = Linear(self.d_model,self.d_model,device = None,dtype = None)    
        self.v_proj = Linear(self.d_model,self.d_model,device = None,dtype = None)    
        #输出投影层，Linear
        self.output_proj = Linear(self.d_model,self.d_model,device = None,dtype = None)
        #Rope初始化
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k , max_seq_len, device = device)
        else:
            self.rope = None
        
    
    def forward(self, x: torch.Tensor,token_positions: torch.Tensor = None) ->torch.Tensor:
        b,s,d = x.shape        
        
        #线性 投影，拆分多头 einops.rearrange替代view+transpose
        #d -- > (h d_k),h前移至s前
        q = einops.rearrange(self.q_proj(x),"... s (h d) -> ... h s d" , h = self.num_heads)
        k = einops.rearrange(self.k_proj(x),"... s (h d) -> ... h s d" , h = self.num_heads)
        v = einops.rearrange(self.v_proj(x),"... s (h d) -> ... h s d" , h = self.num_heads)
        #应用旋转编码Rope
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(s,decive = x.device).expand(b,s)
            
            q = self.rope(q,token_positions)
            k = self.rope(k,token_positions)
        
        #生成因果掩码（下三角矩阵） s* s torch.tril(,dtype = torch.bool)
        mask = torch.tril(torch.ones(s,s,device = x.device,dtype = torch.bool))
        
        #核心注意力计算 --->（batch，heads,seq,d_k）
        attn_out = scaled_dot_product_attention(q,k,v,mask)
        
        #合并多头 ---》 (batch,seq,d)
        attn_out = rearrange(attn_out,"... h s d -> ... s (h d)")
        return self.output_proj(attn_out)
    