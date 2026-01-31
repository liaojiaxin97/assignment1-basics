import torch
import torch.nn as nn
class Linear(nn.Module):
    """
        Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
        """
    def __init__(self,in_features:int ,out_features:int,device = None,dtype = None):
        #正确初始化父类的内部状态（如参数、子模块的注册等）
        super().__init__()
        self.in_feature  = in_features
        self.out_feature = out_features
    
        factory_kwargs = {'device':device,"dtype":dtype}
        #

        self.weight = nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        std = (2/ (self.in_feature + self.out_feature) )** 0.5
        #权重初始化(截断正态分布初始化)-->保证 输出方差一致性，确保新号传到深层
        torch.nn.init.trunc_normal_(self.weight,mean = 0,std = std,a = -3 * std,b = 3 * std)



        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        return torch.einsum("...i,oi -> ...o",x,self.weight)
        
        