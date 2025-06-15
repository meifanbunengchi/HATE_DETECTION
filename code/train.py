# train.py
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, EarlyStoppingCallback
from labels import id2label, label2id
from data_preprocess import make_dataset
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from datasets import Dataset

# 加载模型和 tokenizer
MODEL_PATH = "./roberta-wwm"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH, num_labels=len(label2id), id2label=id2label, label2id=label2id)

train_dataset = Dataset.from_list(make_dataset("data/train_fixed.json", tokenizer))
eval_dataset = Dataset.from_list(make_dataset("data/dev_fixed.json", tokenizer))

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    true_preds, true_labels = [], []

    for pred, label in zip(preds, labels):
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                true_preds.append(p_)
                true_labels.append(l_)

    pr, re, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average="macro")
    return {"precision": pr, "recall": re, "f1": f1}

training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()