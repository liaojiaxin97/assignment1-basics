import regex as re
from collections.abc import Iterable

class BPETokenizer:
    
    """
    将任意字符编码为整数ID序列，并能将其还原
    采用字节级处理，确保不会出现未知词(OOV)错误
    """
    def __init__(self, vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]],
                 special_tokens:list[str]):
        #词汇ID双向映射
        self.vocab = vocab
        self.id_to_bytes = vocab
        #vocab.items()：迭代的是 (key, value)。表达式 {v: u for u, v in vocab.items()} 会生成 {value: key}，即把 bytes 映射回其真实的 token_id，适合作为 bytes_to_id。
        self.bytes_to_id = {v:u for u,v in vocab.items()}
        
        #enumerate(vocab)：迭代的是“字典的键”。得到 (index, key)，index 是按插入顺序的序号，与真实 token_id 无关。
        #字典结构为{(byte_a,byte_b):顺序索引}
        self.merges = {pair:i for i,pair in enumerate(merges)}
        
        self.special_token = special_tokens or []
        
        
        #构建特殊token的正则表达式
        
        if self.special_token:
            #优先匹配最长，防止最长中出现部分内容也为特殊token，导致重复匹配
            sorted_special = sorted(self.special_token,key = len ,reverse = True)
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None
        
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        #gpt2_pat = re.compile(PAT)
    def encode(self,text:str) -> list[int]:
        """
        将输入原始字符编码为整数ID列表
        参数：text,需要原始编码的字符串(例如"hello<|end|>world")
        返回：list[int]:编码后的整数
        """
        #边界情况检查
        if not text:
            return []
        
        if not self.special_regex:
            return self._encode_text_segment(text)
        
        tokens = []
        last_pos  = 0
        #---分词---
        #finditer提供了match.start()和match.end()
        #可以了解特殊标记在哪里开始和结束
        for match in self.special_regex.finditer(text):  
            
            #1) findall：只返回匹配到的字符串列表，没有位置信息/# --> ['<endoftext>', '<pad>']  与compile一起用
            #2) finditer：返回可迭代的 Match 对象，能拿到索引 --> [('<endoftext>', 5, 17), ('<pad>', 22, 27)] 通常在迭代时使用

            
            pre_text = text[last_pos : match.start()]
            
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
                
            #match.group()拿到被识别出来的特殊标记字符串
            special_tok = match.group()
            
            tokens.append(self.bytes_to_id[special_tok.encode("utf-8")])
            
            last_pos = match.end()
        #剩余文本要在循环外，保证文本有特殊词汇的预设，但是文本中实际没有特殊词汇的情况
        remaining_text = text[last_pos:]
        if remaining_text:
            tokens.extend(self._encode_text_segment(remaining_text))   
        
        return tokens
    def _encode_text_segment(self,text:str) -> list[int]:
        """
        对不含特殊token的纯文本片段应用BPE合并逻辑
        """
        ids = []
        
        #gpt2预分词,"hello word" --> "hello","world"
        pre_tokens = self.gpt2_pat.findall(text)
        
        for p_tok in pre_tokens:
            #1.单词转字节"hello" --> b"h",b"e",……
            byte_parts = [bytes([b]) for b in p_tok.encode("utf-8")]
            #2. 反复合并
            while len(byte_parts) >=2 :

                best_pair = None
                min_rank = float('inf')
                #a 在所有相邻对中(merge)寻找rank最低(频率最高)的一对
                for i in range(len(byte_parts) - 1):
                    pair = (byte_parts[i],byte_parts[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                if best_pair is None:
                    break
                #b 执行合并
                
                new_byte_pairs = []
                i = 0
                
                while i < len(byte_parts):
                    if i<len(byte_parts) - 1 and (byte_parts[i] == best_pair[0] and byte_parts[i+1] == best_pair[1]):
                        #确保添加到列表的为b"ewe"的形式 而不是(b"e",b"w")的元组形式，方便后续融合
                        new_byte_pairs.append(best_pair[0]+best_pair[1])
                        i = i + 2
                    else:
                        new_byte_pairs.append(byte_parts[i])
                        i = i + 1
                byte_parts = new_byte_pairs
            for part in byte_parts:
                ids.append(self.bytes_to_id[part])            
        return ids
    def decode(self,ids:list[int]) -> str :
        """
        将ID列表解码为原始字符串
        """
        #拆id表找对应词汇
        byte_segments = [self.id_to_bytes[i] for i in ids]
        #按照b作为分隔拼接，拼接为完整的句子
        full_bytes = b"".join(byte_segments)
        #BPE可能会生成不完整字节序列
        #例，中文往往是三个字节，有时只有两个字节，那么解码时会报错，errors保证插入替换符而不报错
        return full_bytes.decode("utf-8",errors="replace")
    def encode_iterable(self,iterable:Iterable[str]) -> Iterable[int]:
        
        for chunk in iterable:
            yield from self.encode(chunk)