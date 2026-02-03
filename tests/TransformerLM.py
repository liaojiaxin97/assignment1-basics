import torch
import torch.nn as nn
import einops
from tests.Embedding import Embedding
from tests.transformer_block import TransformerBlock
from tests.RMSNorm import RMSNorm
from tests.SwiGLu import SwiGLU
from tests.attention import softmax
from tests.linear import Linear
from tests.topPfilter import _top_p_filter
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
    @torch.no_grad()

    def generate(self,
                prompt_ids: torch.Tensor,
                max_new_tokens: int,
                eos_token_id, int = None,
                temperature: float = 1.0,
                top_p: float = 1.0
                ) -> torch.Tensor:
        """
        从模型生成文本ID序列
        
        参数：
            提示词ID
            最多生成的词数
            停止生成的tokenid
            温度系数（越高越随机，越低越确定）
            he采样阈值
        """
        
        #设置为评估模式
        self.eval()
        
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            
            #1. 裁剪输入： 模型只能处理 context_length长度的内容
            # 如果生成的序列过长，只取最后的 context_length个词
            #idx_cond(batch, 上下文长度)，只取context_length长度的最近的单词
            idx_cond = generated[:, -self.context_length: ]
            
            #2. 前向传播得到Logit
            #推理只关心最后一步，
            logtis = self.forward(idx_cond) #(batch,T,Vocab)
            logits = logits[:, -1, :] #(batch,Vocab)
            
            #3. 应用trick
            #3.1 temperature
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8)
            
            #3.2 top-p
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)
            
            #4. 归一化采样
            probs = softmax(logits, dim = -1)
            #执行的是加权随机采样（Weighted Sampling）,根据概率大小进行随机抽选
            #num_samples=1只需要抽1 个结果
            next_token = torch.multinomial(probs, num_samples=1) #(Batch,1)
            
            #5. 拼接新词
            generated = torch.cat((generated, next_token), dim = 1)
            
            #
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

            
            
        