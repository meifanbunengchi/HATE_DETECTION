# fix_json.py
import json

with open("data/dev.json", "r", encoding="utf-8") as fin:
    data = json.load(fin)

with open("data/dev_fixed.json", "w", encoding="utf-8") as fout:
    for item in data:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("已转换为 JSONL 格式：data/train_fixed.json")