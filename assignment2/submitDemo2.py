import submitit
import time
def add(a, b):
    time.sleep(4) # 模拟耗时
    return a + b
params = [(5, 7), (10, 20), (100, 200)]# 准备多组参数
# 一句话提交一个任务数组！
# 1. 实例化一个“助理”（Executor）
# AutoExecutor 会自动检测到你正在 Slurm 环境下，并进行配置
executor = submitit.AutoExecutor(folder="slurm_logs")
jobs = executor.map_array(add, *zip(*params)) # zip(*params) 会把 [(5,7),...] 变成
([5,10,100], [7,20,200]) 
print(f"一次性提交了 {len(jobs)} 个任务！")
# 等待所有任务完成，并一次性取回所有结果
results = [job.result() for job in jobs]
print(f"所有任务都完成了，结果是: {results}")
# 输出将会是: 所有任务都完成了，结果是: [12, 30, 300]