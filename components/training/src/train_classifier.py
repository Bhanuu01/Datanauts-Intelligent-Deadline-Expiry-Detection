import json, os, time, shutil, tempfile, platform, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F, mlflow, mlflow.pytorch
import numpy as np
from collections import Counter
from datasets import load_from_disk
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score as skf1, accuracy_score, classification_report

os.environ["AWS_ACCESS_KEY_ID"]      = os.getenv("AWS_ACCESS_KEY_ID", "datanauts-key")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.getenv("AWS_SECRET_ACCESS_KEY", "datanauts-secret")
os.environ["GIT_PYTHON_REFRESH"]     = "quiet"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
EXPERIMENT = "deadline-detection-classifier"
DATA_PATH  = "./data/deadline_sentences"
OUTPUT_DIR = "/tmp/deadline-clf"
REPO_ROOT  = Path(__file__).resolve().parents[3]

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
}


def _env_int(name, default=0):
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def maybe_limit_split(ds, env_var):
    limit = _env_int(env_var, 0)
    if limit > 0 and len(ds) > limit:
        return ds.select(range(limit))
    return ds


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


def load_classifier_data(none_ratio, seed=42):
    dd       = load_from_disk(DATA_PATH)
    train_ds = maybe_limit_split(downsample_none(dd["train"], none_ratio, seed), "BOOTSTRAP_MAX_TRAIN_SAMPLES")
    val_ds   = maybe_limit_split(dd["val"], "BOOTSTRAP_MAX_VAL_SAMPLES")
    test_ds  = maybe_limit_split(dd["test"], "BOOTSTRAP_MAX_TEST_SAMPLES")

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} sentences")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        c = Counter([int(x) for x in ds["classifier_label"]])
        print(f"  [{name}] none={c[0]}  expiration={c[1]}  effective={c[2]}  renewal={c[3]}")

    return train_ds, val_ds, test_ds


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

    remove = [c for c in train_ds.column_names if c != "classifier_label"]
    tok_train = train_ds.map(tokenize, batched=True, remove_columns=remove).rename_column("classifier_label", "label")
    tok_val   = val_ds.map(tokenize,   batched=True, remove_columns=remove).rename_column("classifier_label", "label")
    tok_test  = test_ds.map(tokenize,  batched=True, remove_columns=remove).rename_column("classifier_label", "label")

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
        warmup_steps=50,
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
    model_dir = Path(f"{OUTPUT_DIR}-{model_name}")
    trainer.save_model(str(model_dir))
    for checkpoint_dir in model_dir.glob("checkpoint-*"):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
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
        active_run = mlflow.active_run()
        mlflow.log_params({
            "model_name":     model_name,
            "base_model":     cfg["base_model"],
            "task":           "sequence_classification",
            "num_labels":     NUM_LABELS,
            "label_schema":   "none|expiration|effective|renewal|agreement|notice_period",
            "epochs":         cfg["epochs"],
            "batch_size":     cfg["batch_size"],
            "learning_rate":  cfg["learning_rate"],
            "max_seq_length": cfg["max_seq_length"],
            "fp16":           cfg["fp16"],
            "none_ratio":     cfg["none_ratio"],
            "focal_loss":     cfg.get("focal", False),
            "train_size":     len(train_ds),
            "val_size":       len(val_ds),
            "test_size":      len(test_ds),
            "gpu":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_vram_gb":    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
            "pytorch":        torch.__version__,
            "platform":       platform.platform(),
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
            "focal_loss":          cfg.get("focal", False),
        })
        if report:
            mlflow.log_text(report, "clf_classification_report.txt")
        model_dir = Path(f"{OUTPUT_DIR}-{model_name}")
        if model_dir.is_dir():
            metadata = {
                "experiment": EXPERIMENT,
                "run_id": active_run.info.run_id if active_run else None,
                "run_name": model_name,
                "status": "FINISHED",
                "metrics": {
                    "test_f1": test_result.get("eval_f1", 0),
                    "test_accuracy": test_result.get("eval_accuracy", 0),
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
                        help="Which classifier variant to run")
    args = parser.parse_args()

    set_seeds(42)
    cfg = dict(CONFIGS[args.model])
    cfg["base_model"] = resolve_base_model(
        cfg["base_model"], "CLASSIFIER_BASE_MODEL_PATH", "deadline-clf-roberta_clf_v6"
    )
    epoch_override = _env_int("TRAIN_EPOCH_OVERRIDE", 0)
    if epoch_override > 0:
        cfg["epochs"] = epoch_override
    train_ds, val_ds, test_ds = load_classifier_data(cfg["none_ratio"])

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
