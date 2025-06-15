# eval.py
import json
import torch
import difflib
from transformers import BertTokenizerFast, BertForTokenClassification
from labels import id2label, label2id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "./checkpoints"  # 最佳模型保存路径
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# BIO拆分为四元组候选
def extract_entities(text, tokens, labels):
    spans = {"Target": [], "Argument": [], "Group": [], "Hate": []}
    current = None
    start = -1

    for i, label in enumerate(labels):
        tag = id2label[label]
        if tag == "O" or tag == "-100":
            if current:
                word = ''.join(tokens[start:i])
                spans[current].append(word)
                current = None
            continue
        pos = tag.split("-")[1]
        if tag.startswith("B-"):
            if current:
                word = ''.join(tokens[start:i])
                spans[current].append(word)
            current = pos
            start = i
        elif tag.startswith("I-") and pos == current:
            continue
        else:
            if current:
                word = ''.join(tokens[start:i])
                spans[current].append(word)
            current = None

    if current:
        spans[current].append(''.join(tokens[start:]))

    return spans

# 将实体拼接组合为四元组
def make_quadruples(spans):
    quadruples = []
    if not spans["Target"] and not spans["Argument"]:
        return []
    targets = spans["Target"] or ["NULL"]
    arguments = spans["Argument"] or ["NULL"]
    groups = spans["Group"] or ["non-hate"]
    hateful = "hate" if spans["Hate"] else "non-hate"

    for t in targets:
        for a in arguments:
            for g in groups:
                quadruples.append(f"{t} | {a} | {g} | {hateful}")
    return quadruples

# 输出格式处理
def format_output(quads):
    return " [SEP] ".join(quads) + " [END]"

# difflib 相似度
def string_sim(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

# 软匹配逻辑
def soft_match(pred_quads, gold_quads):
    tp = 0
    for p in pred_quads:
        p_parts = [s.strip() for s in p.split("|")]
        for g in gold_quads:
            g_parts = [s.strip() for s in g.split("|")]
            if p_parts[2] == g_parts[2] and p_parts[3] == g_parts[3]:
                sim_target = string_sim(p_parts[0], g_parts[0])
                sim_argument = string_sim(p_parts[1], g_parts[1])
                if sim_target > 0.5 and sim_argument > 0.5:
                    tp += 1
                    break
    return tp

# 主评估函数
def evaluate():
    data = load_data("data/dev.json")
    gold_lines = []
    pred_lines = []

    with open("demo.txt", "w", encoding='utf-8') as fout:
        for sample in data:
            text = sample["content"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy().tolist()
            tokens = tokenizer.tokenize(text)
            pred_labels = preds[1:len(tokens)+1]  # Skip [CLS], keep only token-aligned

            spans = extract_entities(text, tokens, pred_labels)
            quads = make_quadruples(spans)
            predict_str = format_output(quads)
            fout.write(predict_str + "\n")

            pred_lines.append(predict_str.strip())
            gold_lines.append(sample["output"].strip())

    # 开始评分
    hard_tp, soft_tp = 0, 0
    total_pred, total_gold = 0, 0

    for pred, gold in zip(pred_lines, gold_lines):
        pred_quads = [p.strip() for p in pred.replace("[END]", "").split("[SEP]")]
        gold_quads = [g.strip() for g in gold.replace("[END]", "").split("[SEP]")]
        pred_set = set(pred_quads)
        gold_set = set(gold_quads)

        hard_tp += len(set(pred_set) & set(gold_set))
        soft_tp += soft_match(pred_quads, gold_quads)

        total_pred += len(pred_set)
        total_gold += len(gold_set)

    def f1(tp, pred_n, gold_n):
        p = tp / (pred_n + 1e-8)
        r = tp / (gold_n + 1e-8)
        return 2 * p * r / (p + r + 1e-8)

    f1_hard = f1(hard_tp, total_pred, total_gold)
    f1_soft = f1(soft_tp, total_pred, total_gold)

    print("\n✅ 验证集评估结果：")
    print(f"Hard Match F1: {f1_hard:.4f}")
    print(f"Soft Match F1: {f1_soft:.4f}")
    print(f"Average F1: {(f1_hard + f1_soft)/2:.4f}")

# 运行验证
if __name__ == "__main__":
    evaluate()