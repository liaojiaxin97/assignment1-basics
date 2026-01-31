
import torch
from collections.abc import Iterable

def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter],max_norm:float):
    """
    全局梯度裁剪
    """
    
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if not params_with_grad:
        return 
    
    # 计算全局L2 范数
    
    total_norm = 0.0
    
    for p in params_with_grad:
        #使用.detach()很重要
        
        #梯度裁剪是在计算完导数后进行的数值操作，我们不希望“计算范数”的过程也被计入计算图
        
        ##torch.norm(..., p =2 )算出当前层梯度的L2范数L_i
        param_norm = torch.norm(p.grad.detach(),p = 2)
        #将各层范数的平方累加(L_total = sqrt(sum(L_i^2)))
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 检查是否触发裁剪
    eps = 1e-6
    if total_norm > max_norm:
        
        clip_coef = max_norm/(total_norm + eps)
        # inplace 修改每个参数的梯度
        #使用mul_直接修改内存，不产生临时副本，节省显存
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)