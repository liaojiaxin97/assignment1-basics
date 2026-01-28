from pathlib import Path
import json
from pathlib import Path
p = Path("/root/Desktop/assignment1-basics/train_BPE/output/TinyStoriesV2-GPT4-trainvocab.json")
obj = json.loads(p.read_text(encoding="utf-8"))
best_tid, best_tok = max(obj.items(), key=lambda kv: (len(kv[1]), kv[1]))
print(len(best_tok), best_tok, best_tid)