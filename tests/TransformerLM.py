import torch
import torch.nn as nn
import einops
from tests.Embedding import Embedding
from tests.transformer_block import TransformerBlock
from tests.RMSNorm import RMSNorm
from tests.SwiGLu import SwiGLU
from tests.attention import softmax
from tests.linear import Linear
#        vocab_size=vocab_size,
        # context_length=n_keys,
        # d_model=d_model,
        # num_layers=n_layers,
        # num_heads=n_heads,
        # d_ff=d_ff,
        # rope_theta=theta,
        # weights=state_dict,
        # in_indices=in_indices,
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, 
                 num_layers: int, num_head: int, d_ff: int, rope_theta: float, 
                 device = None, dtype = None,
                 #新的实验参数
                 user_rms_norm: bool = True,
                 norm_mode: str = "pre",
                 ffn_type: str = "swiglu"):
        
        super().__init__()
        self.vocab_size = vocab_size
        #determine the dimension of embedding matrix
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_head = num_head
        self.d_ff = d_ff
        #embedding_dim跟后续的block维度对齐
        self.Embedding = Embedding(self.vocab_size, self.d_model, device=device, dtype=dtype)
        self.TransformerBlockList = nn.ModuleList()
        for _ in range(num_layers):
            self.TransformerBlock = TransformerBlock(self.d_model,
                                            self.num_head, self.d_ff,
                                            self.max_seq_len,rope_theta,
                                            user_rms_norm = user_rms_norm,
                                            norm_mode = norm_mode,
                                            ffn_type = ffn_type)
            
            self.TransformerBlockList.append(self.TransformerBlock)
        if user_rms_norm:
            self.ln_final = RMSNorm(d_model=self.d_model)
        else:
            self.ln_final = nn.Identity()
        self.lm_head = Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)
        
    def forward(self,token_ids):
        
        b,s = token_ids.shape
        
        #位置信息用于位置编码
        #(s,) -> (1,s) -> (b,s)
        token_positions = torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)
        
        x = self.Embedding(token_ids)
        
        for block in self.TransformerBlockList:
            x = block(x,token_positions)
        #归一化
        x = self.ln_final(x)
        #投影 (……,d_model) --> (……,vocabsize)
        return self.lm_head(x)
        