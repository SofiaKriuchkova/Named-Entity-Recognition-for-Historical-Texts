#!/usr/bin/env python3
"""train_ner.py

Train a HuggingFace token-classification model on CoNLL/BIO TSV data.

Expected input format (TSV):
token <TAB> label
Sentences separated by blank lines.

Example:
Tartu   B-LOC
linn    O

...

Outputs:
- trained model + tokenizer in --output_dir
- metrics JSON in --output_dir/metrics.json
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Optional but recommended for NER scoring
try:
    from nervaluate import Evaluator
except Exception:  # pragma: no cover
    Evaluator = None


def read_conll_tsv(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Reads a CoNLL-style TSV file: token\tlabel, blank line between sentences."""
    sentences_tokens: List[List[str]] = []
    sentences_labels: List[List[str]] = []
    cur_toks: List[str] = []
    cur_labs: List[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if cur_toks:
                    sentences_tokens.append(cur_toks)
                    sentences_labels.append(cur_labs)
                    cur_toks, cur_labs = [], []
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Bad line in {path}: {line!r} (expected token\tlabel)")
            tok, lab = parts[0], parts[1]
            cur_toks.append(tok)
            cur_labs.append(lab)

    if cur_toks:
        sentences_tokens.append(cur_toks)
        sentences_labels.append(cur_labs)

    return sentences_tokens, sentences_labels


def build_label_maps(all_labels: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted({lab for sent in all_labels for lab in sent})
    label2id = {l: i for i, l in enumerate(uniq)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def make_dataset(tokens: List[List[str]], labels: List[List[str]]) -> Dataset:
    return Dataset.from_dict({"tokens": tokens, "ner_tags": labels})


def tokenize_and_align_labels(examples: Dict[str, Any], tokenizer, label2id: Dict[str, int]):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
    )
    aligned_labels = []
    for i, labs in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        out = []
        for w in word_ids:
            if w is None:
                out.append(-100)
            elif w != prev:
                out.append(label2id[labs[w]])
            else:
                # Subword: keep same label for simplicity; alternatively set -100
                out.append(label2id[labs[w]])
            prev = w
        aligned_labels.append(out)
    tokenized["labels"] = aligned_labels
    return tokenized


def tags_to_entities(tag_seq: List[str]) -> List[Dict[str, str]]:
    """Convert BIO tags to list of {type, start, end} over token indices."""
    entities = []
    start = None
    ent_type = None

    def close(end_idx):
        nonlocal start, ent_type
        if start is not None and ent_type is not None:
            entities.append({"type": ent_type, "start": start, "end": end_idx})
        start, ent_type = None, None

    for i, tag in enumerate(tag_seq):
        if tag == "O" or tag == "":
            close(i)
            continue
        if tag.startswith("B-"):
            close(i)
            ent_type = tag[2:]
            start = i
        elif tag.startswith("I-"):
            t = tag[2:]
            if ent_type is None:
                ent_type = t
                start = i
            elif t != ent_type:
                close(i)
                ent_type = t
                start = i
        else:
            # Unknown scheme: treat as outside
            close(i)

    close(len(tag_seq))
    return entities


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # sentence-level tag sequences (skip -100)
        true_tags = []
        pred_tags = []
        for p_sent, l_sent in zip(preds, labels):
            t, p = [], []
            for pi, li in zip(p_sent, l_sent):
                if li == -100:
                    continue
                t.append(id2label[int(li)])
                p.append(id2label[int(pi)])
            true_tags.append(t)
            pred_tags.append(p)

        if Evaluator is None:
            # Fallback: token-level accuracy
            total = sum(len(x) for x in true_tags)
            correct = sum(sum(1 for a, b in zip(t, p) if a == b) for t, p in zip(true_tags, pred_tags))
            return {"token_accuracy": correct / total if total else 0.0}

        # nervaluate expects entity spans
        y_true = [tags_to_entities(seq) for seq in true_tags]
        y_pred = [tags_to_entities(seq) for seq in pred_tags]
        evaluator = Evaluator(y_true, y_pred, tags=None)  # tags inferred from spans
        results, _ = evaluator.evaluate()

        # Use "strict" overall if present; else approximate
        overall = results.get("strict", results.get("ent_type", {})).get("overall", {})
        return {
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
            "f1": float(overall.get("f1", 0.0)),
        }

    return compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="HF model id, e.g. tartuNLP/EstBERT or EMBEDDIA/est-roberta")
    ap.add_argument("--train", required=True, help="Path to train TSV")
    ap.add_argument("--dev", required=True, help="Path to dev TSV")
    ap.add_argument("--test", required=False, help="Optional test TSV (will be evaluated after training)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--max_steps", type=int, default=-1, help="Use for quick runs; -1 means full epochs")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_DISABLED", "true")
    set_seed(args.seed)

    tr_tok, tr_lab = read_conll_tsv(args.train)
    dv_tok, dv_lab = read_conll_tsv(args.dev)

    label2id, id2label = build_label_maps(tr_lab + dv_lab)

    ds = DatasetDict({
        "train": make_dataset(tr_tok, tr_lab),
        "validation": make_dataset(dv_tok, dv_lab),
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    tokenized = ds.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        max_steps=args.max_steps,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_out = {"dev": {k: float(v) for k, v in eval_metrics.items()}}
    if args.test:
        ts_tok, ts_lab = read_conll_tsv(args.test)
        test_ds = make_dataset(ts_tok, ts_lab).map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id), batched=True)
        test_metrics = trainer.evaluate(test_ds)
        metrics_out["test"] = {k: float(v) for k, v in test_metrics.items()}

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    print("Saved model and metrics to:", args.output_dir)


if __name__ == "__main__":
    main()
