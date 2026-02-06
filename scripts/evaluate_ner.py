#!/usr/bin/env python3
"""evaluate_ner.py

Evaluate a saved token-classification model checkpoint on a CoNLL/BIO TSV test file.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments

try:
    from nervaluate import Evaluator
except Exception:  # pragma: no cover
    Evaluator = None


def read_conll_tsv(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    sentences_tokens, sentences_labels = [], []
    cur_toks, cur_labs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_toks:
                    sentences_tokens.append(cur_toks); sentences_labels.append(cur_labs)
                    cur_toks, cur_labs = [], []
                continue
            tok, lab = line.split("\t")[:2]
            cur_toks.append(tok); cur_labs.append(lab)
    if cur_toks:
        sentences_tokens.append(cur_toks); sentences_labels.append(cur_labs)
    return sentences_tokens, sentences_labels


def tags_to_entities(tag_seq):
    entities=[]
    start=None; ent_type=None
    def close(end):
        nonlocal start, ent_type
        if start is not None and ent_type is not None:
            entities.append({"type": ent_type, "start": start, "end": end})
        start=None; ent_type=None
    for i,tag in enumerate(tag_seq):
        if tag=="O" or tag=="":
            close(i); continue
        if tag.startswith("B-"):
            close(i); ent_type=tag[2:]; start=i
        elif tag.startswith("I-"):
            t=tag[2:]
            if ent_type is None:
                ent_type=t; start=i
            elif t!=ent_type:
                close(i); ent_type=t; start=i
        else:
            close(i)
    close(len(tag_seq))
    return entities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory with saved model+tokenizer")
    ap.add_argument("--test", required=True, help="Test TSV")
    ap.add_argument("--out", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_DISABLED", "true")

    tokens, labels = read_conll_tsv(args.test)
    ds = Dataset.from_dict({"tokens": tokens, "ner_tags": labels})

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    label2id = model.config.label2id
    id2label = {int(k): v for k, v in model.config.id2label.items()} if isinstance(model.config.id2label, dict) else model.config.id2label

    def tokenize_and_align(examples):
        tok = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True)
        aligned=[]
        for i,labs in enumerate(examples["ner_tags"]):
            word_ids = tok.word_ids(batch_index=i)
            prev=None; out=[]
            for w in word_ids:
                if w is None:
                    out.append(-100)
                elif w!=prev:
                    out.append(label2id[labs[w]])
                else:
                    out.append(label2id[labs[w]])
                prev=w
            aligned.append(out)
        tok["labels"]=aligned
        return tok

    tokenized = ds.map(tokenize_and_align, batched=True)
    collator = DataCollatorForTokenClassification(tokenizer)

    def compute_metrics(eval_pred):
        logits, lab_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        true_tags=[]; pred_tags=[]
        for ps, ls in zip(preds, lab_ids):
            t=[]; p=[]
            for pi, li in zip(ps, ls):
                if li == -100:
                    continue
                t.append(id2label[int(li)])
                p.append(id2label[int(pi)])
            true_tags.append(t); pred_tags.append(p)

        if Evaluator is None:
            total=sum(len(x) for x in true_tags)
            correct=sum(sum(1 for a,b in zip(t,p) if a==b) for t,p in zip(true_tags,pred_tags))
            return {"token_accuracy": correct/total if total else 0.0}

        y_true=[tags_to_entities(seq) for seq in true_tags]
        y_pred=[tags_to_entities(seq) for seq in pred_tags]
        evaluator=Evaluator(y_true,y_pred,tags=None)
        results,_=evaluator.evaluate()
        overall = results.get("strict", results.get("ent_type", {})).get("overall", {})
        return {
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
            "f1": float(overall.get("f1", 0.0)),
        }

    ta = TrainingArguments(output_dir=os.path.join(args.model_dir, "_eval"), per_device_eval_batch_size=8, report_to=[])
    trainer = Trainer(model=model, args=ta, eval_dataset=tokenized, tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics)

    metrics = trainer.evaluate()
    metrics = {k: float(v) for k, v in metrics.items()}
    out_path = args.out or os.path.join(args.model_dir, "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved:", out_path)
    print(metrics)


if __name__ == "__main__":
    main()
