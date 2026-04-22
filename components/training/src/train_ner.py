import json, os, re, time, shutil, tempfile, platform, argparse, random
import torch, torch.nn as nn, mlflow, mlflow.pytorch
import numpy as np
from collections import Counter
from datasets import load_from_disk
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

os.environ["AWS_ACCESS_KEY_ID"]      = os.getenv("AWS_ACCESS_KEY_ID", "datanauts-key")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.getenv("AWS_SECRET_ACCESS_KEY", "datanauts-secret")
os.environ["GIT_PYTHON_REFRESH"]     = "quiet"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://129.114.27.190:30900")

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://129.114.27.190:30500")
EXPERIMENT  = "deadline-detection-ner"
DATA_PATH   = "./data/deadline_sentences"
OUTPUT_DIR  = "/tmp/deadline-ner"

LABEL_LIST  = [
    "O",
    "B-EXP_DATE",    "I-EXP_DATE",
    "B-START_DATE",  "I-START_DATE",
    "B-DURATION",    "I-DURATION",
    "B-NOTICE_DATE", "I-NOTICE_DATE",
]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL    = {i: l for i, l in enumerate(LABEL_LIST)}

MONTHS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
}

CONFIGS = {
    "baseline": {
        "base_model": None, "epochs": 0, "learning_rate": 0,
        "batch_size": 16, "max_seq_length": 256, "fp16": False,
    },
    "bert_ner_v1": {
        "base_model": "dslim/bert-base-NER", "epochs": 3, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True,
    },
    "bert_ner_v2": {
        "base_model": "dslim/bert-base-NER", "epochs": 3, "learning_rate": 5e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True,
    },
    "bert_ner_v3": {
        "base_model": "dslim/bert-base-NER", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True,
    },
    "bert_ner_v4": {
        "base_model": "dslim/bert-base-NER", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True,
    },
    "bert_ner_v5": {
        "base_model": "dslim/bert-base-NER", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True,
    },
    "bert_base_cased": {
        "base_model": "bert-base-cased", "epochs": 3, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True,
    },
}

# O-tag weight=1.0, entity tags upweighted — amplifies rare entity signal
# (downweighting O caused false-positive explosion: recall=1.0, precision=0.01)
# NOTICE_DATE weighted higher (12x) — rarest entity in training data
NER_WEIGHTS = torch.tensor([1.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 12.0, 12.0])


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Weighted NER Trainer — down-weights O tokens to reduce label dominance ─────────
class WeightedNERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        w       = NER_WEIGHTS.to(logits.device)
        loss    = nn.CrossEntropyLoss(weight=w, ignore_index=-100)(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def load_ner_data():
    dd        = load_from_disk(DATA_PATH)
    # Train on ALL sentences: model needs O-token context to avoid false positives
    # (non-none only = 550 sentences → model tags everything as entity)
    train_ds  = dd["train"].select_columns(["tokens", "ner_tags"])
    val_ds    = dd["val"].select_columns(["tokens", "ner_tags"])
    test_ds   = dd["test"].select_columns(["tokens", "ner_tags"])
    print(f"Train (all): {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} sentences")
    return train_ds, val_ds, test_ds


def tokenize_dataset(train_ds, val_ds, test_ds, tokenizer, max_length):
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"], truncation=True,
            max_length=max_length, is_split_into_words=True,
        )
        aligned = []
        for i, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            prev, ids = None, []
            for wid in word_ids:
                if wid is None:     ids.append(-100)
                elif wid != prev:   ids.append(labels[wid])
                else:               ids.append(-100)
                prev = wid
            aligned.append(ids)
        tokenized["labels"] = aligned
        return tokenized

    tok_train = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
    tok_val   = val_ds.map(tokenize_and_align,   batched=True, remove_columns=val_ds.column_names)
    tok_test  = test_ds.map(tokenize_and_align,  batched=True, remove_columns=test_ds.column_names)
    return tok_train, tok_val, tok_test


def compute_metrics(p):
    preds, labels = p
    preds       = np.argmax(preds, axis=2)
    true_preds  = [[ID2LABEL[x] for x, l in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    true_labels = [[ID2LABEL[l] for l in lab if l != -100] for lab in labels]
    report  = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
    metrics = {
        "f1":        f1_score(true_labels,        true_preds, zero_division=0),
        "precision": precision_score(true_labels, true_preds, zero_division=0),
        "recall":    recall_score(true_labels,    true_preds, zero_division=0),
    }
    for entity in ["EXP_DATE", "START_DATE", "DURATION", "NOTICE_DATE"]:
        if entity in report:
            metrics[f"{entity}_f1"]       = report[entity]["f1-score"]
            metrics[f"{entity}_precision"] = report[entity]["precision"]
            metrics[f"{entity}_recall"]    = report[entity]["recall"]
    return metrics


def run_baseline(test_ds):
    true_seqs, pred_seqs = [], []
    for sample in test_ds:
        tokens   = sample["tokens"]
        tags     = sample["ner_tags"]
        true_seq = [ID2LABEL[t] for t in tags]
        pred_seq, i = [], 0
        while i < len(tokens):
            tok_clean = tokens[i].rstrip(".,").lower()
            if tok_clean in MONTHS:
                pred_seq.append("B-EXP_DATE")
                i += 1
                while i < len(tokens) and re.match(r"^\d{1,4}[,.]?$", tokens[i]):
                    pred_seq.append("I-EXP_DATE")
                    i += 1
            else:
                pred_seq.append("O")
                i += 1
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)
    return {
        "entity_f1":        f1_score(true_seqs, pred_seqs),
        "entity_precision": precision_score(true_seqs, pred_seqs),
        "entity_recall":    recall_score(true_seqs, pred_seqs),
    }


def train_model(model_name, cfg, tok_train, tok_val, tok_test, tokenizer):
    model  = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"], num_labels=len(LABEL_LIST),
        id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
    )
    t_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}-{model_name}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        warmup_steps=50,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
    )
    trainer = WeightedNERTrainer(
        model=model, args=t_args,
        train_dataset=tok_train, eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(f"Training {model_name} | lr={cfg['learning_rate']} | epochs={cfg['epochs']}...")
    t0           = time.time()
    train_result = trainer.train()
    model_dir = Path(f"{OUTPUT_DIR}-{model_name}")
    trainer.save_model(str(model_dir))
    for checkpoint_dir in model_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    train_time   = time.time() - t0

    print("Evaluating on test set...")
    preds_out   = trainer.predict(tok_test)
    raw_preds   = np.argmax(preds_out.predictions, axis=2)
    raw_labs    = preds_out.label_ids
    true_preds  = [[ID2LABEL[x] for x, l in zip(p, lb) if l != -100] for p, lb in zip(raw_preds, raw_labs)]
    true_labels = [[ID2LABEL[l] for l in lb if l != -100] for lb in raw_labs]
    report      = classification_report(true_labels, true_preds)
    print(report)
    test_result = {
        "eval_f1":        f1_score(true_labels, true_preds),
        "eval_precision": precision_score(true_labels, true_preds),
        "eval_recall":    recall_score(true_labels, true_preds),
        "eval_loss":      preds_out.metrics.get("test_loss", 0),
    }
    print(f"Done! Time: {train_time:.1f}s | Test F1: {test_result['eval_f1']:.4f}")
    return trainer.model, train_result, test_result, train_time, report


def log_to_mlflow(model_name, cfg, model, train_result, test_result, train_time,
                  train_ds, val_ds, test_ds, report=""):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=model_name):
        active_run = mlflow.active_run()
        mlflow.log_params({
            "model_name":     model_name,
            "base_model":     cfg.get("base_model", "regex"),
            "epochs":         cfg["epochs"],
            "batch_size":     cfg["batch_size"],
            "learning_rate":  cfg["learning_rate"],
            "max_seq_length": cfg["max_seq_length"],
            "fp16":           cfg["fp16"],
            "num_labels":     len(LABEL_LIST),
            "label_schema":   "O|B-EXP_DATE|I-EXP_DATE|B-START_DATE|I-START_DATE|B-DURATION|I-DURATION|B-NOTICE_DATE|I-NOTICE_DATE",
            "train_size":     len(train_ds),
            "val_size":       len(val_ds),
            "test_size":      len(test_ds),
            "gpu":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_vram_gb":    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
            "pytorch":        torch.__version__,
            "platform":       platform.platform(),
        })
        mlflow.log_metrics({
            "total_train_time_sec":  train_time,
            "time_per_epoch_sec":    train_time / max(cfg["epochs"], 1),
            "samples_per_sec":       len(train_ds) * max(cfg["epochs"], 1) / max(train_time, 1),
            "train_loss":            train_result.training_loss if train_result else 0,
            "test_f1":               test_result.get("eval_f1",              test_result.get("entity_f1",        0)),
            "test_precision":        test_result.get("eval_precision",        test_result.get("entity_precision", 0)),
            "test_recall":           test_result.get("eval_recall",           test_result.get("entity_recall",    0)),
            "test_EXP_DATE_f1":      test_result.get("eval_EXP_DATE_f1",      0),
            "test_START_DATE_f1":    test_result.get("eval_START_DATE_f1",    0),
            "test_DURATION_f1":      test_result.get("eval_DURATION_f1",      0),
            "test_NOTICE_DATE_f1":   test_result.get("eval_NOTICE_DATE_f1",   0),
            "test_loss":             test_result.get("eval_loss", 0),
        })
        if report:
            mlflow.log_text(report, "ner_classification_report.txt")
        model_dir = Path(f"{OUTPUT_DIR}-{model_name}")
        if model_dir.is_dir():
            metadata = {
                "experiment": EXPERIMENT,
                "run_id": active_run.info.run_id if active_run else None,
                "run_name": model_name,
                "status": "FINISHED",
                "metrics": {
                    "test_f1": test_result.get("eval_f1", test_result.get("entity_f1", 0)),
                    "test_precision": test_result.get("eval_precision", test_result.get("entity_precision", 0)),
                    "test_recall": test_result.get("eval_recall", test_result.get("entity_recall", 0)),
                },
            }
            metadata_path = model_dir / "mlflow_run.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            with tempfile.TemporaryDirectory() as artifact_dir:
                artifact_root = Path(artifact_dir)
                for artifact_name in [
                    "config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "training_args.bin",
                    "vocab.txt",
                    "merges.txt",
                    "sentencepiece.bpe.model",
                    "spiece.model",
                ]:
                    src = model_dir / artifact_name
                    if src.exists():
                        shutil.copy2(src, artifact_root / artifact_name)
                (artifact_root / "mlflow_run.json").write_text(json.dumps(metadata, indent=2))
                mlflow.log_artifacts(str(artifact_root), artifact_path="model")
            print(f"Model weights uploaded → MinIO (artifact_path=model)")
        print(f"Logged {model_name} → {MLFLOW_URI}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIGS.keys()),
                        help="Which NER model variant to run")
    args = parser.parse_args()

    set_seeds(42)
    cfg = CONFIGS[args.model]
    train_ds, val_ds, test_ds = load_ner_data()

    if args.model == "baseline":
        t0      = time.time()
        metrics = run_baseline(test_ds)
        elapsed = time.time() - t0
        print(f"Baseline F1: {metrics['entity_f1']:.4f} | Time: {elapsed:.2f}s")
        log_to_mlflow(
            "ner_baseline_regex",
            {"base_model": "regex", "epochs": 0, "batch_size": 16,
             "learning_rate": 0, "max_seq_length": 256, "fp16": False},
            None, None,
            {"eval_f1": metrics["entity_f1"],
             "eval_precision": metrics["entity_precision"],
             "eval_recall": metrics["entity_recall"]},
            elapsed, train_ds, val_ds, test_ds,
        )
        return

    tokenizer                    = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok_train, tok_val, tok_test = tokenize_dataset(train_ds, val_ds, test_ds, tokenizer, cfg["max_seq_length"])
    model, train_result, test_result, train_time, report = train_model(
        args.model, cfg, tok_train, tok_val, tok_test, tokenizer,
    )
    log_to_mlflow(args.model, cfg, model, train_result, test_result, train_time,
                  train_ds, val_ds, test_ds, report)


if __name__ == "__main__":
    main()
