import torch
import torch.nn as nn
class Embedding(nn.Module):
    
    def __init__(self,num_embeddings : int , embedding_dim : int , device = None , dtype = None):
        super().__init__()
        factory_kwargs = {'device':device,"dtype":dtype}
        
        #想象成一个有 num_embeddings 行的大表，每一行存储了一个词的 embedding_dim 维特征向量。
        #num_embedding由分词器输出的词汇表数量决定---vocab_size
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))
        #初始化
        nn.init.trunc_normal_(self.weight , mean = 0.0 , std = 1.0  ,a = -3.0 , b = 3.0)
  
        
    def forward(self,token_id:torch.Tensor) ->torch.Tensor:
        #token_id :B S
        #token_id为索引矩阵，存储的是 0 到 9999 之间的整数，每个整数代表一个词在权重表中的行号
        return self.weight[token_id] #B S D
    
    

