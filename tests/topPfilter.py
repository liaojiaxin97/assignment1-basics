import torch
from tests.attention import softmax
def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    内部工具函数：执行top-P截断 ---用作推理
    """
    
    #对词表分值进行降序排序
    #sorted_logits：分数排序
    #sorted_indices： 分数对应的索引
    """
    原始 logits (ID对应: 猫, 狗, 鸟):

    ID 0 (猫): 0.1
    ID 1 (狗): 0.9
    ID 2 (鸟): 0.4
    执行 sort:

    sorted_logits (内容): [0.9, 0.4, 0.1] <-- 顺序是：狗，鸟，猫
    sorted_indices (序号): [1, 2, 0]
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1)
    
    #计算累积概率分布
    #假设概率是 [0.5, 0.3, 0.1, 0.1] --》 [[0.5, 0.8, 0.9, 1.0]]
    cumulative_probs = torch.cumsum(softmax(sorted_logits,dim = -1),dim = -1)
    
    #创建掩码：去掉累积概率超过p的token
    #逻辑：保留最小集合Vp，使其概率之和>=p
    #把所有超过p的位置标记为True（需要移除）
    """
    示例
    #选词概率（排序后）：[0.4, 0.3, 0.2, 0.1]
    # 阈值 Top-P:p = 0.6
    # 累积概率(cumsum):[0.4, 0.7, 0.9, 1.0]
    如果不做移位（直接用 > p)
    代码逻辑:sorted_indices_to_remove = cumulative_probs > 0.6

    第 1 个词 (0.4):0.4 > 0.6? False (保留)
    第 2 个词 (0.7):0.7 > 0.6? True (删除!) <-- 问题在这里
    第 3 个词 (0.9):0.9 > 0.6? True (删除)
    结果：你只保留了第 1 个词（概率 0.4)
    错误：根据 Top-P 定义，我们要保留概率总和 ≥0.6=v(p)的最小集合
    现在只剩 0.4,显然不够。我们需要保留第 2 个词，让总和达到 0.7≥0.6=v(p)
    """

    sorted_indices_to_remove = cumulative_probs > p
    
    #关键修正：确保至少保留第一个词
    #并且要保留第一个使概率超过p的词
    #将标记向右拉一格
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    #将排序后的掩码还原回原始词汇表的顺序。
    indices_to_remove = sorted_indices_to_remove.scatter(dim = 1, index = sorted_indices, src = sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf")) 
    
    return logits