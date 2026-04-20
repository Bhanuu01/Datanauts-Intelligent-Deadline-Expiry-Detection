import os, json, time, platform, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F, mlflow, mlflow.pytorch
import numpy as np
from collections import Counter
from datasets import load_from_disk, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score as skf1, accuracy_score, classification_report
from env_config import setup_env, CFG

setup_env()

MLFLOW_URI = os.environ["MLFLOW_TRACKING_URI"]
EXPERIMENT = "deadline-detection-classifier"
DATA_PATH  = "./data/deadline_sentences"
OUTPUT_DIR = "/tmp/deadline-clf"

CLF_LABEL2ID = {"none": 0, "expiration": 1, "effective": 2, "renewal": 3, "agreement": 4, "notice_period": 5}
CLF_ID2LABEL = {0: "none", 1: "expiration", 2: "effective", 3: "renewal", 4: "agreement", 5: "notice_period"}
NUM_LABELS   = 6

CONFIGS = {
    "baseline": {
        "base_model": "roberta-base", "epochs": 0, "learning_rate": 0,
        "batch_size": 16, "max_seq_length": 256, "fp16": False, "none_ratio": 5, "focal": False,
    },
    "roberta_clf_v1": {
        "base_model": "roberta-base", "epochs": 3, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 5, "focal": False,
    },
    "roberta_clf_v2": {
        "base_model": "roberta-base", "epochs": 3, "learning_rate": 5e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 5, "focal": False,
    },
    "roberta_clf_v3": {
        "base_model": "roberta-base", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 3, "focal": False,
    },
    "roberta_clf_v4": {
        "base_model": "roberta-base", "epochs": 4, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 8, "focal": False,
    },
    "roberta_clf_v5": {
        "base_model": "roberta-base", "epochs": 4, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 8, "focal": False,
    },
    "roberta_clf_v6": {
        "base_model": "roberta-base", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 10, "focal": True,
    },
    "roberta_clf_v7": {
        "base_model": "roberta-base", "epochs": 5, "learning_rate": 2e-5,
        "batch_size": 16, "max_seq_length": 256, "fp16": True, "none_ratio": 10, "focal": True,
    },
}


# ── Weighted CE loss Trainer ─────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        w       = self.class_weights.to(logits.device)
        loss    = nn.CrossEntropyLoss(weight=w)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Weighted Focal loss Trainer — stronger minority class focus ───────────────
class WeightedFocalTrainer(Trainer):
    def __init__(self, class_weights, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.gamma         = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        w       = self.class_weights.to(logits.device)
        ce_loss = nn.CrossEntropyLoss(weight=w, reduction="none")(logits, labels)
        pt      = torch.exp(-ce_loss)
        loss    = ((1 - pt) ** self.gamma * ce_loss).mean()
        return (loss, outputs) if return_outputs else loss


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def downsample_none(ds, none_ratio, seed=42):
    labels   = [int(x) for x in ds["classifier_label"]]
    pos_idx  = [i for i, l in enumerate(labels) if l != 0]
    none_idx = [i for i, l in enumerate(labels) if l == 0]
    max_none = min(len(none_idx), none_ratio * len(pos_idx))
    random.seed(seed)
    keep = sorted(random.sample(none_idx, max_none) + pos_idx)
    return ds.select(keep)


def load_classifier_data(none_ratio, seed=42, extra_data=None):
    dd       = load_from_disk(DATA_PATH)
    train_ds = downsample_none(dd["train"], none_ratio, seed)
    val_ds   = dd["val"]
    test_ds  = dd["test"]

    cfg_extra_size = 0
    if extra_data:
        if not os.path.exists(extra_data):
            print(f"[extra_data] WARNING: path not found: {extra_data} — skipping")
        else:
            with open(extra_data) as f:
                rows = [json.loads(l) for l in f if l.strip()]
            labelled = [r for r in rows if r.get("classifier_label") is not None]
            if labelled:
                extra_ds = Dataset.from_list([
                    {"sentence": r["sentence"], "classifier_label": CLF_LABEL2ID.get(r["classifier_label"], r["classifier_label"])}
                    for r in labelled
                ])
                cfg_extra_size = len(labelled)
                train_ds = concatenate_datasets([train_ds, extra_ds])
                print(f"[extra_data] Appended {len(labelled)} feedback samples → train now {len(train_ds)}")
            else:
                print(f"[extra_data] No labelled entries found in {extra_data}")

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} sentences")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        c = Counter([int(x) for x in ds["classifier_label"]])
        print(f"  [{name}] none={c[0]}  expiration={c[1]}  effective={c[2]}  renewal={c[3]}  agreement={c[4]}  notice_period={c[5]}")

    return train_ds, val_ds, test_ds, cfg_extra_size


def compute_class_weights(train_ds, max_weight=10.0):
    labels  = [int(x) for x in train_ds["classifier_label"]]
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = [
        min(total / (NUM_LABELS * max(counts.get(i, 1), 1)), max_weight)
        for i in range(NUM_LABELS)
    ]
    print(f"Class counts : {dict(counts)}")
    print(f"Class weights: {[f'{w:.2f}' for w in weights]} (capped at {max_weight}x)")
    return torch.tensor(weights, dtype=torch.float)


def tokenize_data(train_ds, val_ds, test_ds, tokenizer, max_length):
    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True,
                         max_length=max_length, padding=False)

    train_remove = [c for c in train_ds.column_names if c != "classifier_label"]
    val_remove   = [c for c in val_ds.column_names   if c != "classifier_label"]
    test_remove  = [c for c in test_ds.column_names  if c != "classifier_label"]
    tok_train = train_ds.map(tokenize, batched=True, remove_columns=train_remove).rename_column("classifier_label", "label")
    tok_val   = val_ds.map(tokenize,   batched=True, remove_columns=val_remove).rename_column("classifier_label", "label")
    tok_test  = test_ds.map(tokenize,  batched=True, remove_columns=test_remove).rename_column("classifier_label", "label")

    tok_train.set_format("torch")
    tok_val.set_format("torch")
    tok_test.set_format("torch")
    return tok_train, tok_val, tok_test


def compute_clf_metrics(p):
    preds  = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    metrics = {
        "accuracy":    float(accuracy_score(labels, preds)),
        "f1":          float(skf1(labels, preds, average="macro",    zero_division=0)),
        "f1_weighted": float(skf1(labels, preds, average="weighted", zero_division=0)),
    }
    for i, name in enumerate(["none", "expiration", "effective", "renewal", "agreement", "notice_period"]):
        metrics[f"f1_{name}"] = float(
            skf1(labels, preds, labels=[i], average="macro", zero_division=0)
        )
    return metrics


def run_baseline(test_ds):
    labels_all     = [int(x) for x in test_ds["classifier_label"]]
    majority       = Counter(labels_all).most_common(1)[0][0]
    preds_majority = [majority] * len(labels_all)
    f1  = skf1(labels_all, preds_majority, average="macro",    zero_division=0)
    acc = accuracy_score(labels_all, preds_majority)
    print(f"Majority class baseline — class: {CLF_ID2LABEL[majority]} | F1: {f1:.4f} | Acc: {acc:.4f}")
    return {"eval_f1": f1, "eval_accuracy": acc, "eval_loss": 0}


def train_classifier(model_name, cfg, tok_train, tok_val, tok_test, tokenizer, class_weights):
    model  = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=NUM_LABELS,
        id2label=CLF_ID2LABEL, label2id=CLF_LABEL2ID,
    )
    t_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}-{model_name}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
    )
    TrainerCls = WeightedFocalTrainer if cfg.get("focal", False) else WeightedTrainer
    focal_kwargs = {"gamma": 2.0} if cfg.get("focal", False) else {}
    trainer = TrainerCls(
        class_weights=class_weights,
        **focal_kwargs,
        model=model, args=t_args,
        train_dataset=tok_train, eval_dataset=tok_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_clf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print(f"Training {model_name} | lr={cfg['learning_rate']} | epochs={cfg['epochs']}...")
    t0           = time.time()
    train_result = trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}-{model_name}")
    train_time   = time.time() - t0

    print("Evaluating on test set...")
    preds_out = trainer.predict(tok_test)
    preds     = np.argmax(preds_out.predictions, axis=1)
    labels    = preds_out.label_ids
    report    = classification_report(labels, preds,
                    target_names=[CLF_ID2LABEL[i] for i in range(NUM_LABELS)],
                    labels=list(range(NUM_LABELS)), zero_division=0)
    print(report)
    test_result = {
        "eval_f1":       float(skf1(labels, preds, average="macro",    zero_division=0)),
        "eval_accuracy": float(accuracy_score(labels, preds)),
        "eval_loss":     preds_out.metrics.get("test_loss", 0),
    }
    print(f"Done! Time: {train_time:.1f}s | Test F1: {test_result['eval_f1']:.4f} | Accuracy: {test_result['eval_accuracy']:.4f}")
    return trainer.model, train_result, test_result, train_time, report


def log_to_mlflow(model_name, cfg, model, train_result, test_result, train_time,
                  train_ds, val_ds, test_ds, report=""):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model_name":        model_name,
            "base_model":        cfg["base_model"],
            "task":              "sequence_classification",
            "num_labels":        NUM_LABELS,
            "label_schema":      "none|expiration|effective|renewal|agreement|notice_period",
            "epochs":            cfg["epochs"],
            "batch_size":        cfg["batch_size"],
            "learning_rate":     cfg["learning_rate"],
            "max_seq_length":    cfg["max_seq_length"],
            "fp16":              cfg["fp16"],
            "none_ratio":        cfg["none_ratio"],
            "focal_loss":        cfg.get("focal", False),
            "train_size":        len(train_ds),
            "val_size":          len(val_ds),
            "test_size":         len(test_ds),
            "extra_data_path":   cfg.get("extra_data") or "none",
            "extra_data_size":   cfg.get("extra_data_size", 0),
            "gpu":               torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_vram_gb":       round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
            "pytorch":           torch.__version__,
            "platform":          platform.platform(),
        })
        mlflow.log_metrics({
            "total_train_time_sec":     train_time,
            "time_per_epoch_sec":       train_time / max(cfg["epochs"], 1),
            "samples_per_sec":          len(train_ds) * max(cfg["epochs"], 1) / max(train_time, 1),
            "train_loss":               train_result.training_loss if train_result else 0,
            "test_f1":                  test_result.get("eval_f1", 0),
            "test_f1_none":             test_result.get("eval_f1_none", 0),
            "test_f1_expiration":       test_result.get("eval_f1_expiration", 0),
            "test_f1_effective":        test_result.get("eval_f1_effective", 0),
            "test_f1_renewal":          test_result.get("eval_f1_renewal", 0),
            "test_f1_agreement":        test_result.get("eval_f1_agreement", 0),
            "test_f1_notice_period":    test_result.get("eval_f1_notice_period", 0),
            "test_accuracy":            test_result.get("eval_accuracy", 0),
            "test_loss":                test_result.get("eval_loss", 0),
        })
        train_counts = Counter(int(x) for x in train_ds["classifier_label"])
        mlflow.log_params({
            "train_none":          train_counts.get(0, 0),
            "train_expiration":    train_counts.get(1, 0),
            "train_effective":     train_counts.get(2, 0),
            "train_renewal":       train_counts.get(3, 0),
            "train_agreement":     train_counts.get(4, 0),
            "train_notice_period": train_counts.get(5, 0),
        })
        if report:
            mlflow.log_text(report, "clf_classification_report.txt")
        model_dir = f"{OUTPUT_DIR}-{model_name}"
        if os.path.isdir(model_dir):
            mlflow.log_artifacts(model_dir, artifact_path="model")
            print(f"Model weights uploaded → MinIO (artifact_path=model)")
            try:
                run_id = mlflow.active_run().info.run_id
                mlflow.register_model(f"runs:/{run_id}/model", model_name)
                print(f"Model registered in MLflow Registry as '{model_name}'")
            except Exception as reg_err:
                print(f"[WARN] Model Registry unavailable: {reg_err}")
        print(f"Logged {model_name} → {MLFLOW_URI}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CONFIGS.keys()),
                        help="Which classifier variant to run")
    parser.add_argument("--extra_data", default=None,
                        help="Optional path to feedback JSONL to concatenate with train split")
    args = parser.parse_args()

    set_seeds(42)
    cfg = dict(CONFIGS[args.model])
    cfg["extra_data"] = args.extra_data
    train_ds, val_ds, test_ds, extra_data_size = load_classifier_data(cfg["none_ratio"], extra_data=args.extra_data)
    cfg["extra_data_size"] = extra_data_size

    if args.model == "baseline":
        t0          = time.time()
        test_result = run_baseline(test_ds)
        elapsed     = time.time() - t0
        log_to_mlflow("clf_baseline_majority", {**cfg, "epochs": 0}, None, None,
                      test_result, elapsed, train_ds, val_ds, test_ds)
        return

    tokenizer                    = AutoTokenizer.from_pretrained(cfg["base_model"])
    class_weights                = compute_class_weights(train_ds)
    tok_train, tok_val, tok_test = tokenize_data(train_ds, val_ds, test_ds, tokenizer, cfg["max_seq_length"])
    model, train_result, test_result, train_time, report = train_classifier(
        args.model, cfg, tok_train, tok_val, tok_test, tokenizer, class_weights,
    )
    log_to_mlflow(args.model, cfg, model, train_result, test_result, train_time,
                  train_ds, val_ds, test_ds, report)


if __name__ == "__main__":
    main()
