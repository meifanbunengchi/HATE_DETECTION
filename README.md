# HATE_DETECTION
hate detection
 仇恨言论识别与四元组抽取系统 | Hate Speech Quadruple Extraction
本项目旨在识别中文文本中的仇恨言论，通过命名实体识别方法抽取四元组：

Target | Argument | Group | Hateful

基于预训练语言模型（如 BERT 或 RoBERTa）进行微调训练，识别出包含攻击对象、论点、所指群体以及是否为仇恨内容的结构化结果。

📂 项目结构
.

├── 2.py                     # 将 JSON 格式转换为按行 JSONL 格式

├── data_preprocess.py       # 数据处理与 BIO 标签生成

├── eval.py                  # 模型评估（硬匹配 & 软匹配）

├── train.py                 # 模型训练脚本（基于 Trainer API）

├── labels.py                # BIO 标签映射 + 设备信息

├── predict_test.py          # 单独推理入口，支持经验规则增强结果

├── data/

│  
 ├── train_fixed.json     # JSONL格式训练数据

│   
└── dev_fixed.json       # JSONL格式验证数据

├── test1.json               # 测试用例（数组形式）

└── demomin.txt              # 推理结果输出

1️⃣ 安装依赖
pip install torch transformers datasets scikit-learn

2️⃣ 数据预处理
原始 JSON → JSONL 转换（逐行格式）
python 2.py
输出文件: data/dev_fixed.json

3️⃣ 训练模型（NER）
python train.py
使用 roberta-wwm 模型训练并自动保存最优模型到 ./checkpoints

4️⃣ 验证模型效果
python eval.py

输出示例：

✅ 验证集评估结果：
Hard Match F1: 0.7312
Soft Match F1: 0.7981
Average F1: 0.7647
5️⃣ 推理预测（支持经验规则增强）
先准备测试数据格式 test1.json：

[
  {
    "content": "这些黑人整天无所事事。"
  },
  ...
]

然后运行：

python predict_test.py

将结果输出到 demomin.txt，每行一个预测输出：

黑人 | 无所事事 | Racism | hate [END]
🔍 四元组定义说明
组件	含义说明
Target	被评论或攻击的对象（如“黑人”）
Argument	对对象的描述、指责、情绪成分等
Group	所属群体，如种族、地域、性别等
Hateful	是否为仇恨（hate / non-hate）
🏷️ 标签体系（标签文件）
BIO 序列标注标签定义如下：

O                  ⟶ 非实体
B-Target, I-Target ⟶ 评论对象
B-Argument, I-Argument ⟶ 论点
B-Group, I-Group   ⟶ 所指群体（地域/性别等）
B-Hate, I-Hate     ⟶ 是否具有攻击性
转换使用（见 labels.py）：

label2id = {   # tag -> idx
    "O": 0,
    "B-Target": 1,
    "I-Target": 2,
    ...
}
id2label = {v: k for k, v in label2id.items()}

📈 模型评估说明
使用两种不同的匹配策略进行性能评价：

✅ 硬匹配（Hard Match）
预测四元组 与 真实四元组 完全一致，才算正确
🟡 软匹配（Soft Match）
只需 Group 和 Hateful 完全一致，且 Target 和 Argument 字符相似度 > 0.5
使用 difflib.SequenceMatcher 库计算相似度
🧠 推理增强（predict_test.py）
预测时支持词典规则补充识别能力：

包含多个仇恨触发词列表，如：
RACISM_TRIGGERS：黑人、黑鬼、劣种...
REGION_TRIGGERS：河南、东北、南方人...
GENDER_TRIGGERS：女拳、母猪、男的都...
在模型未输出仇恨/群体预测时，自动通过规则尝试补充。

📌 模型可自由替换
你可以将 train.py 和 predict_test.py 中的预训练模型路径修改为：

MODEL_PATH = "bert-base-chinese"
或：www模型路径

支持以下中文模型：

hfl/chinese-roberta-wwm-ext
hfl/chinese-macbert-base
自训练模型权重路径
📄 示例输入 & 输出
输入内容：

{
  "content": "这些黑皮猴只会打游戏，懒得要命。"
}

模型输出：

黑皮猴 | 懒得要命 | Racism | hate [END]
