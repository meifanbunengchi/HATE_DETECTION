# labels.py
labels = [
    "O",
    "B-Target", "I-Target",
    "B-Argument", "I-Argument",
    "B-Group", "I-Group",
    "B-Hate", "I-Hate"
]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}