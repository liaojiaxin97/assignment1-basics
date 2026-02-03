import numpy as np
import os

data_path = "data/TinyStoriesV2-GPT4-train.bin"

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
else:
    # 读取数据（假设是 uint16）
    data = np.fromfile(data_path, dtype=np.uint16)
    
    max_id = data.max()
    min_id = data.min()
    
    print(f"File: {data_path}")
    print(f"Total tokens: {len(data)}")
    print(f"Max Token ID: {max_id}")
    print(f"Min Token ID: {min_id}")
    
    # 检查是否越界
    VOCAB_SIZE = 10000
    if max_id >= VOCAB_SIZE:
        print(f"\n❌ CRITICAL ISSUE: Max ID ({max_id}) >= Vocab Size ({VOCAB_SIZE})")
        print("This will definitely cause 'CUDA error: device-side assert triggered' in Embedding layer.")
    else:
        print(f"\n✅ Data looks safe for Vocab Size {VOCAB_SIZE}")
