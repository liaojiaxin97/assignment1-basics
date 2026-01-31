

import torch
import typing
import os
def save_checkpoint(
    model: torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration: int,
    out: typing.Union[str,os.PathLike, typing.BinaryIO , typing.IO[bytes]],
):
    checkpoint = {
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration
        
    }
    
    torch.save(checkpoint,out)


def load_checkpoint(
    src:typing.Union[str,os.PathLike, typing.BinaryIO , typing.IO[bytes]],
    model:torch.nn.Module,
    optimizer: torch.optim.Optimizer
)->int:
    
    # 1. 加载字典
    checkpoint = torch.load(src, map_location="cpu")
    
    # 2. 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 3. 恢复优化器
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint['iteration']