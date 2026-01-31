import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    max_seq_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    """
    随机采样一个训练批次
    
    return：
        x: [batchsize,max_seq_length]
        y: [batchsize,max_seq_length]
    #memmap不真正把数据读入ram，而是在磁盘和逻辑内存间建立映射
    #当使用get_batch时，操作系统才会取磁盘上精确地把那几kb的数据拉进缓存
    #data = np.memp("train.bin",dtype = np.uint16,mode = "r")
    """
    n = len(dataset)
    #总长度-最大句子长度 = 输入x能用的最大起点
    #总长度-最大句子长度-1 = 输入y能用的最大起点
    max_idx = n - max_seq_length - 1
    
    #随机选择batch_size个起点
    #max_idx是能取的点，但（0,max_idx）取不到，所以使用max_idx+1
    ix = torch.randint(0 , max_idx + 1, (batch_size,))
    
    x = torch.stack([
        torch.from_numpy(dataset[i:i+max_seq_length].astype(np.int64)) for i in ix
    ])
        
    y = torch.stack([
        torch.from_numpy(dataset[i+1: i + 1 + max_seq_length].astype(np.int64)) for i in ix
    ])
    
    return x.to(device),y.to(device)