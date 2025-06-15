# data_preprocess.py
import json
from transformers import BertTokenizerFast
from labels import label2id

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def find_span(text, phrase):
    start = text.find(phrase)
    if start == -1:
        return None
    return start, start + len(phrase)

def parse_output(output_str):
    items = output_str.strip().split('[SEP]')
    spans = []
    for item in items:
        item = item.replace('[END]', '').strip()
        if item:
            parts = [x.strip() for x in item.split('|')]
            if len(parts) == 4:
                spans.append({
                    'target': parts[0],
                    'argument': parts[1],
                    'group': parts[2],
                    'hateful': parts[3]
                })
    return spans

def get_bio_labels(tokens, offsets, spans, text):
    labels = ['O'] * len(tokens)

    def mark_span(span_text, label_prefix):
        span = find_span(text, span_text)
        if span:
            start, end = span
            for i, (s_tok, e_tok) in enumerate(offsets):
                if s_tok >= end:
                    break
                if e_tok <= start:
                    continue
                if s_tok >= start and e_tok <= end:
                    if labels[i] == 'O':
                        labels[i] = f"B-{label_prefix}" if s_tok == start else f"I-{label_prefix}"

    for item in spans:
        mark_span(item['target'], 'Target')
        mark_span(item['argument'], 'Argument')
        if item['group'] != 'non-hate':
            mark_span(item['group'], 'Group')
        if item['hateful'] == 'hate':
            mark_span(item['argument'], 'Hate')  # hate标记在Argument上

    return [label2id.get(lbl, 0) for lbl in labels]

def make_dataset(file_path, tokenizer):
    raw_data = load_data(file_path)
    dataset = []

    for sample in raw_data:
        text = sample['content']
        spans = parse_output(sample.get('output', ''))

        encoded = tokenizer(text, truncation=True, return_offsets_mapping=True)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        offsets = encoded['offset_mapping'][1:-1]  # remove CLS / SEP

        tokens = tokenizer.tokenize(text)
        labels = get_bio_labels(tokens, offsets, spans, text)

        labels = [label2id['O']] + labels[:len(input_ids) - 2] + [label2id['O']]
        labels += [-100] * (len(input_ids) - len(labels))

        dataset.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })

    return dataset