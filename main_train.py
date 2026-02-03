import argparse
import os 
import torch
import numpy as np

import wandb 

from cs336_basics.TransformerLM import TransformerLM
from cs336_basics.AdamW import AdamW
from cs336_basics.gradientClipping import clip_gradient_norm
from cs336_basics.learningRateSchedule import get_lr_cosine_schedule
from cs336_basics.getBatch import get_batch
from cs336_basics.checkPointing import save_checkpoint,load_checkpoint
from cs336_basics.crossEntropy import cross_entropy

def main():
    parser = argparse.ArgumentParser()
    #模型基础超参数
    parser.add_argument("--batch_size", type= int,default=32)
    parser.add_argument("--context_length", type= int,default=256)
    parser.add_argument("--d_model", type= int,default=512)
    parser.add_argument("--num_layers", type= int,default=4)
    parser.add_argument("--num_heads", type= int,default=8)
    parser.add_argument("--d_ff", type= int,default=2048)
    parser.add_argument("--vocab_size", type= int,default=10000)
    
    
    #实验/消融开关
    
    #移除RMSNorm
    parser.add_argument("--no_rms_norm", action = "store_true", help= "Disable RMSNorm completely")
    #Pre-norm vs Post-norm
    parser.add_argument("--norm_mode", type = str, default="pre",choices=["pre","post"],help = "Normalization Placement")
    #移除Rope
    parser.add_argument("--no_rope", action= "store_true", help = "Disable Rotary Positional Embeddings")
    #SwiGLU vsSiLU
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices= ["swiglu","silu"], help = "type of Feed-Forward Network")
    
    #优化器超参数
    parser.add_argument("--lr", type= float,default=6e-4)
    parser.add_argument("--max_iters", type= int,default=10000)
    parser.add_argument("--warmup_iters", type= int,default=1000)
    parser.add_argument("--min_lr", type= float,default=6e-5)
    parser.add_argument("--max_norm", type= float,default=1.0)
    
    #路径与系统
    parser.add_argument("--train_data_path", type= str, required=True)
    parser.add_argument("--valid_data_path", type= str, required=True)
    parser.add_argument("--out_dir", type= str,default="out")
    parser.add_argument("--device", type= str,default="cuda" if torch.cuda.is_available() else "cpu")
    
    #wandb 设置
    parser.add_argument("--run_name", type=str, default=None, help = "wandb实验名称")
    
    args = parser.parse_args()
    #exist_ok = True 目录偌存在，不会报错
    os.makedirs(args.out_dir, exist_ok= True)
    
    
    #1. 加载数据(memmap)
    #uint16存储的二进制文件
    
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"training data not found at {args.train_data_path}")
    
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"validtion data not found at {args.valid_data_path}")    
    
    #np. memmap 延迟加载数据到内存，适合大数据集，并将二进制文件转为dtype(uint16)数组
    
    train_data = np.memmap(args.train_data_path,dtype = np.uint16, mode = "r")
    val_data = np.memmap(args.valid_data_path,dtype = np.uint16, mode = "r")
    
    print(f"训练集大小{len(train_data)} tokens")
    print(f"验证集大小{len(val_data)} tokens")

    #2. 处理消融实验逻辑
    
    actual_rope_theta = None if args.no_rope else 10000.0
    #use_rms_norm
    use_rms_norm = not args.no_rms_norm
    
    #3. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model = args.d_model,
        num_layers= args.num_layers,
        num_head = args.num_heads,
        d_ff = args.d_ff,
        rope_theta= actual_rope_theta,
        device= args.device,
        #实验参数
        user_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.device)
    
    print(f"Model config: Norm = {args.norm_mode}, UseNorm = {use_rms_norm}, FFN = {args.ffn_type}, Rope= {not args.no_rope}")
    
    
    #4. 优化器初始化
    optimizer = AdamW(model.parameters(),lr = args.lr, weight_decay=0.1)
    
    #5. 检查点恢复逻辑
    
    start_iter = 0
    #负责把目录路径（args.out_dir）和文件名（"ckpt.pt"）拼在一起，生成一个完整的文件路径
    #若 args.out_dir 是 "out", 那么 ckpt_path 变量的值就是 "out/ckpt.pt"。
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path,model,optimizer)
        print(f"Resuming from iteration {start_iter}")
    
    
    #6. 初始化wandb监控
    
    wandb.init(
        project = "cs336-assignment1",
        name = args.run_name,
        config = args
    )
    
    #7. 主训练循环
    
    for it in range(start_iter, args.max_iters):
        #A. 学习率更新
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        #B. 训练步骤
        
        model.train()
        x,y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        logits = model(x)
        
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        clip_gradient_norm(model.parameters(), args.max_norm)
        
        optimizer.step()
        
        #C.验证与日志记录
        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                #loss是tensor，包括额外信息如梯度函数 grad_fn、设备信息等
                #loss.item() 会提取张量里的标量值，把它变成一个标准的 Python 浮点数（float）
                print(f"Iter {it}: train_loss {loss.item(): 4f}, val_loss{v_loss.item(): 4f}, lr {lr:.2e}")
                               
                wandb.log ({
                    "train/loss" : loss.item(),
                    "val/loss": v_loss.item(),
                    "lr": lr,
                    "iter": it + 1
                })
        if it % 1000 ==0 and it > 0:
            save_checkpoint(model,optimizer,it,ckpt_path)
    save_checkpoint(model,optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))

    wandb.finish()
    
if  __name__ == "__main__":
    main()