
import os
import json
import pathlib
from functools import lru_cache
FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


def bytes_to_unicode():
    """
    创建一个从 0..255 字节到可见 Unicode 字符的映射。
    优先使用可打印 ASCII(33..126) 和 Latin-1 可见区(161..172, 174..255)，
    其余字节映射到 >=256 的码位，保证可逆。
    """
    # 可见集合
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    # 补全剩余未覆盖的字节
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 转为字符
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))
 
  
def save_tokenizer_files(vocab, merges, out_dir,dataset):
    os.makedirs(out_dir, exist_ok=True)

    # 初始化映射表
    byte_encoder = bytes_to_unicode()

    # 词表保存（使用 byte_encoder 将 bytes 转换为可见字符串）
    json_vocab = {
        k: "".join(byte_encoder[b] for b in v)
        for k, v in vocab.items()
    }
    with open(os.path.join(out_dir, dataset+"vocab.json"), "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, ensure_ascii=False, indent=4)

    # 合并规则保存
    with open(os.path.join(out_dir, dataset+"merges.txt"), "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")



import json
import time
import resource
from adapters import run_train_bpe

def main():
    dataset  = "owt_train"
    input_path = "/root/Desktop/assignment1-basics/train_BPE/data/" + (dataset + ".txt")
    vocab_size = 320000
    
    special_tokens = ["<endoftext>"]
    
    outpput_dir = outpput_dir = pathlib.Path(__file__).resolve().parent / "output"
    
    print(f"开始训练bpe分词，词表大小: {vocab_size}")
    
    vocab,merges = run_train_bpe(input_path,vocab_size,special_tokens)
    
    t0 = time.perf_counter()
    vocab, merges = run_train_bpe(input_path, vocab_size, special_tokens)
    dt = time.perf_counter() - t0

    # ru_maxrss: Linux 为 KB（近似），统一显示为 MiB
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak_kb / 1024.0

    print(f"训练耗时: {dt:.2f}s, 进程峰值内存: {peak_mib:.1f} MiB")
    
    save_tokenizer_files(vocab,merges,outpput_dir,dataset)
    

if __name__ == "__main__":
    main()