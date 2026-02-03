import torch
import torch.nn as nn
# 确认你的环境支持 CUDA
if not torch.cuda.is_available():
    print("CUDA is not available. This script requires a GPU.")
else:
    device = torch.device("cuda")
# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=1024, out_features=2048)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=2048, out_features=512)
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))
# 2. 实例化模型和输入数据，并移动到 GPU
model = SimpleModel().to(device)
# 创建一个需要梯度的输入张量
input_tensor = torch.randn(256, 1024, device=device, requires_grad=True)
# 3. 开始记录内存历史
print("======== 开始内存性能分析 ========")

torch.cuda.memory._record_memory_history(max_entries=1000000)
# --- 我们想要分析的核心代码段 ---
# 执行一次前向传播
output = model(input_tensor)
# 计算一个标量损失
loss = output.sum()
# 执行一次反向传播
loss.backward()
# ---------------------------------
# 4. 保存内存快照到文件
snapshot_filename = "simple_model_snapshot.pickle"
torch.cuda.memory._dump_snapshot(snapshot_filename)
# 5. 停止记录
torch.cuda.memory._record_memory_history(enabled=None)
print(f"======== 性能分析结束, 快照已保存至 '{snapshot_filename}' ========")
print("请访问 <https://pytorch.org/memory_viz> 并上传该文件进行分析。")