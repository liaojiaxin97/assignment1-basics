
import math

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) ->float:
    """
    计算it次迭代时，带预热的余弦退火学习率
    """
    
    #1. 预热阶段： 线性增长逻辑
    if it < warmup_iters:
        #从0匀速增长到max_learning_rate
        return max_learning_rate *it / warmup_iters
    
    #2. 衰减周期后：维持最小值
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    #3. 计算当前处于退火阶段的进度 百分比
    # it - warmup_iters :距离预热结束走了多少步
    # cosine_cycle_iters - warmup_iters：整个退火阶段的总长度
    decay_ratio = (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)
    
    #4. 计算余弦函数
    #进度为0时，cos0 =1
    #进度为1时，结果为cos pi = -1
    coeff = 0.5 *(1 + math.cos(math.pi*decay_ratio))
    #学习率从max降向 min
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
    