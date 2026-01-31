
import torch
import math 
from torch.optim import Optimizer
from collections.abc import Iterable

class AdamW(Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9,0.999), eps = 1e-8, weight_decay = 0.01):
        
        #1. 基本参数检查
        if lr < 0.0:
            raise ValueError(f"invalid learning rate:{lr}")
        if not 0.0 <= betas[0] <1.0:
            raise ValueError(f"invalid beta parameter at index 0 : {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon value:{eps}")
        
        defaults = dict(lr = lr ,betas = betas, eps = eps, weight_decay = weight_decay)
        #告诉 PyTorch 的 Optimizer 基类：“帮我把这些参数（params）管理起来，
        # 并且给它们设定好默认的超参数配置（defaults）
        super().__init__(params, defaults)
        
    #优化器的 step 阶段是利用已经计算好的梯度（p.grad）来更新模型参数（p.data）。
    #这个“参数更新”过程本身（例如做减法 w = w - lr * grad）是不需要被记录下来用于反向传播的
    @torch.no_grad()
    def step(self):
        """执行单步优化更新"""
        
        loss = None
        # # group 是一个字典，比如 {'params': [w1, w2, b1], 'lr': 0.001, ...}
        for group in self.param_groups:
            beta1,beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]
            #group["params"]是“一袋子参数”和它们的配置（学习率等）。
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                #在 state = self.state[p] 这一步，p 仅仅是作为“索引键”（Key）或者说“身份证”，用来查找。
                #p 是一个 Parameter 张量对象。PyTorch 使用这个对象的**内存地址（引用）**作为唯一的 Key（键）。
                # 它并不把 p 的 权重数值 复制进去。
                state = self.state[p]
                
                # 3. 状态初始化
                
                if len(state) == 0:
                    state["step"] = 0
                    
                    #m:一阶矩 梯度指数的移动平均
                    state['exp_avg'] = torch.zeros_like(p,
                                                        memory_format=torch.preserve_format)
                    #v:二阶ju 梯度平方的移动平均
                    state['exp_avg_sq'] = torch.zeros_like(p,
                                                        memory_format=torch.preserve_format)                  
                    
                exp_avg,exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                
                t = state["step"]
                
                # 4. 更新ju估计
                
                #m = beta1 *m + (1 - beta1) * g
                #exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                #v = beta2 *v + (1 - beta2) * g*g
                #exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * pow(grad,2)   
                #上述写法只是赋值，没有修改state字典中所存储的对应内容
                #alpha: 缩放系数，避免你要先写一句 grad * (1 - beta1) 产生一个临时张量，然后再去相加
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                
                #addcmul (不带下划线) 是 Out-of-place 操作。它计算结果并返回一个新的 Tensor，但不会修改调用它的那个 Tensor (exp_avg_sq)。
                #addcmul_(带下划线) 是 In-place 操作，直接修改了 exp_avg_sq 的内存
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value = 1-beta2)
                
                # 5 . 计算偏差矫正后的学习率 alpha_t
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)
                
                # 6 . 更新参数：the = theta - alpha_t * m / (sqrt(v) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                #p.addcdiv_(t1,t2,value = 1.0) --> p = p + value * (t1/t2)
                p.addcdiv_(exp_avg, denom, value = -step_size)
                
                # 7 . 应用解耦的权重衰减
                #theta = theta - alpha * lambda * theta
                #p.add_(other,alpha = 1.0) p = p+(alpha * othrt)
                if wd!=0:
                    p.add_(p, alpha = -lr * wd)
        return loss           
                             