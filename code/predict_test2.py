# predict_test.py

import json
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from labels import id2label
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 配置修改区 ====
MODEL_PATH = "D:\en模型\model_checkpoints\checkpoints\checkpoint-2000"
TEST_PATH = "test1.json"
OUTPUT_PATH = "demomin.txt"

# ==== 扩展的仇恨触发词表 ====
RACISM_TRIGGERS = ["黑人", "黑鬼", "黑皮", "劣种", "低人种", "黄皮猴", "黄种人", "白左", "洋奴", "非洲","非默", "黑","犹太","猩猩","嘿人"]
REGION_TRIGGERS = ["河南", "广州", "上海", "东北", "福建", "湖北", "河北", "新疆", "北方", "安徽","广西", "南方", "青海", "江南", "山西", "陕", "甘肃", "黑龙江", "辽宁", "吉林"]
GENDER_TRIGGERS = ["女拳", "女权癌", "母猪", "幕刃", "男人", "国女", "国铝", "不配结婚", "绿茶", "不配生育","小仙女", "女人", "普信男", "女", "男", "妹", "龟男", "舔狗", "女犬", "拳"]
LGBTQ_TRIGGERS = ["基佬", "同",  "gay"]
OTHERS_TRIGGERS = ["时代"]

HATE_TRIGGER_WORDS = (
    RACISM_TRIGGERS + REGION_TRIGGERS + GENDER_TRIGGERS + LGBTQ_TRIGGERS + OTHERS_TRIGGERS
)

GROUP_HINTS = {
    "Racism": RACISM_TRIGGERS,
    "Region": REGION_TRIGGERS,
    "Sexism": GENDER_TRIGGERS,
    "LGBTQ": LGBTQ_TRIGGERS,
    "others": OTHERS_TRIGGERS
}

# ==== 加载模型 ====
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()


# ==== 工具函数 ====
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # JSON array 格式


def extract_spans(tokens, labels):
    spans = {"Target": [], "Argument": [], "Group": [], "Hate": []}
    current_type = None
    current_tokens = []

    for tok, lab in zip(tokens, labels):
        label = id2label.get(lab, "O")
        if label == "O":
            if current_type:
                spans[current_type].append("".join(current_tokens))
                current_type = None
                current_tokens = []
        elif label.startswith("B-"):
            if current_type:
                spans[current_type].append("".join(current_tokens))
            current_type = label[2:]
            current_tokens = [tok]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(tok)
        else:
            current_type = None
            current_tokens = []

    if current_type and current_tokens:
        spans[current_type].append("".join(current_tokens))
    return spans


def postprocess(spans, text):
    # 默认值
    hate = "hate" if spans["Hate"] else "non-hate"
    group = spans["Group"] or []

    # === 🆕 评论对象（Target） + 论点（Argument）分析 ===
    targets = spans.get("Target", [])
    arguments = spans.get("Argument", [])

    # 拼接后判定是否匹配触发词
    combined_texts = []
    for t in targets:
        for a in arguments:
            combined_texts.append(t + a)
            combined_texts.append(a + t)  # 避免错位顺序

    # === 🔍 Hate 判断强化 ===
    if hate == "non-hate":  # 模型未判断为 hate
        strings_to_check = [text] + combined_texts
        for s in strings_to_check:
            if any(word in s for word in HATE_TRIGGER_WORDS):
                hate = "hate"
                break

    # === 🔍 Group 分析强化 ===
    if not group:
        strings_to_check = [text] + combined_texts
        for s in strings_to_check:
            for cat, keywords in GROUP_HINTS.items():
                if any(word in s for word in keywords):
                    group = [cat]
                    break
            if group:
                break
        if not group:
            group = ["non-hate"]

    return hate, group


def combine_to_quadruples(spans, hate, group):
    targets = spans["Target"] or ["NULL"]
    arguments = spans["Argument"] or ["NULL"]
    results = []

    for t in targets:
        for a in arguments:
            for g in group:
                results.append(f"{t} | {a} | {g} | {hate}")
    return results


def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.tokenize(text)
    pred_labels = preds[1:len(tokens)+1]
    spans = extract_spans(tokens, pred_labels)
    hate_flag, group_list = postprocess(spans, text)
    quads = combine_to_quadruples(spans, hate_flag, group_list)
    return quads


# ==== 主程序 ====
if __name__ == "__main__":

    test_data = load_data(TEST_PATH)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for sample in test_data:
            text = sample["content"]
            quads = predict(text)
            output_line = " [SEP] ".join(quads) + " [END]"
            fout.write(output_line + "\n")
            print(f"[输入] {text}")
            print(f"[输出] {output_line}\n")

    print(f"✅ 已完成预测，输出文件：{OUTPUT_PATH}")