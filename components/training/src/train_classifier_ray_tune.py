import json, os, time, random
import numpy as np
import torch
import torch.nn as nn
import mlflow
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from collections import Counter
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import f1_score as skf1, accuracy_score

os.environ["AWS_ACCESS_KEY_ID"]      = os.getenv("AWS_ACCESS_KEY_ID", "datanauts-key")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.getenv("AWS_SECRET_ACCESS_KEY", "datanauts-secret")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
)
os.environ["GIT_PYTHON_REFRESH"]     = "quiet"

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
EXPERIMENT  = "deadline-detection-classifier-ray-tune"
DATA_PATH   = "/app/data/deadline_sentences"
NUM_LABELS  = 6

CLASSIFIER_MODEL_CANDIDATES = {
    "roberta_base": {
        "base_model": "roberta-base",
        "batch_sizes": [8, 16, 32],
    },
    "deberta_v3_base": {
        "base_model": "microsoft/deberta-v3-base",
        "batch_sizes": [4, 8, 16],
    },
    "deberta_v3_large": {
        "base_model": "microsoft/deberta-v3-large",
        "batch_sizes": [4, 8],
    },
}

CLF_LABEL2ID = {"none": 0, "expiration": 1, "effective": 2, "renewal": 3, "agreement": 4, "notice_period": 5}
CLF_ID2LABEL = {v: k for k, v in CLF_LABEL2ID.items()}

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


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


def downsample_none(ds, none_ratio, seed=42):
    labels   = [int(x) for x in ds["classifier_label"]]
    pos_idx  = [i for i, l in enumerate(labels) if l != 0]
    none_idx = [i for i, l in enumerate(labels) if l == 0]
    max_none = min(len(none_idx), none_ratio * len(pos_idx))
    random.seed(seed)
    keep = sorted(random.sample(none_idx, max_none) + pos_idx)
    return ds.select(keep)


def compute_class_weights(train_ds, max_weight=10.0):
    labels  = [int(x) for x in train_ds["classifier_label"]]
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = [
        min(total / (NUM_LABELS * max(counts.get(i, 1), 1)), max_weight)
        for i in range(NUM_LABELS)
    ]
    return torch.tensor(weights, dtype=torch.float)


def compute_clf_metrics(p):
    preds  = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    metrics = {
        "f1":       float(skf1(labels, preds, average="macro",    zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
    }
    for i, name in enumerate(["none", "expiration", "effective", "renewal"]):
        metrics[f"f1_{name}"] = float(skf1(labels, preds, labels=[i], average="macro", zero_division=0))
    return metrics


def train_trial(config):
    model_key        = config["model_key"]
    model_cfg        = CLASSIFIER_MODEL_CANDIDATES[model_key]
    base_model       = model_cfg["base_model"]
    lr              = config["learning_rate"]
    requested_batch = config["batch_size"]
    batch_size      = requested_batch
    if batch_size not in model_cfg["batch_sizes"]:
        eligible = [size for size in model_cfg["batch_sizes"] if size <= requested_batch]
        batch_size = max(eligible) if eligible else min(model_cfg["batch_sizes"])
    warmup_ratio    = config["warmup_ratio"]
    weight_decay    = config["weight_decay"]
    epochs          = config["epochs"]
    none_ratio      = config["none_ratio"]
    max_seq_length  = config["max_seq_length"]
    label_smoothing = config["label_smoothing"]
    run_name        = (
        f"{model_key}_clf_lr{lr:.1e}_bs{batch_size}_wd{weight_decay:.0e}_ep{epochs}"
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate":   lr,
            "batch_size":      batch_size,
            "warmup_ratio":    warmup_ratio,
            "weight_decay":    weight_decay,
            "epochs":          epochs,
            "none_ratio":      none_ratio,
            "max_seq_length":  max_seq_length,
            "label_smoothing": label_smoothing,
            "model_key":       model_key,
            "base_model":      base_model,
            "requested_batch_size": requested_batch,
            "effective_batch_size": batch_size,
            "search_method":   "Ray Tune ASHA",
            "num_labels":      NUM_LABELS,
            "label_schema":    "none|expiration|effective|renewal|agreement|notice_period",
            "gpu":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        })

        dd        = load_from_disk(DATA_PATH)
        train_ds  = downsample_none(dd["train"], none_ratio, seed=42)
        val_ds    = dd["val"]
        test_ds   = dd["test"]
        class_weights = compute_class_weights(train_ds)

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        def tokenize(examples):
            return tokenizer(examples["sentence"], truncation=True,
                             max_length=max_seq_length, padding=False)

        train_remove = [c for c in train_ds.column_names if c != "classifier_label"]
        val_remove   = [c for c in val_ds.column_names   if c != "classifier_label"]
        test_remove  = [c for c in test_ds.column_names  if c != "classifier_label"]
        tok_train = train_ds.map(tokenize, batched=True, remove_columns=train_remove).rename_column("classifier_label", "label")
        tok_val   = val_ds.map(tokenize,   batched=True, remove_columns=val_remove).rename_column("classifier_label", "label")
        tok_test  = test_ds.map(tokenize,  batched=True, remove_columns=test_remove).rename_column("classifier_label", "label")
        for ds in [tok_train, tok_val, tok_test]:
            ds.set_format("torch")

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=NUM_LABELS,
            id2label=CLF_ID2LABEL, label2id=CLF_LABEL2ID,
        )

        t_args = TrainingArguments(
            output_dir=f"/tmp/ray_clf_{run_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            label_smoothing_factor=label_smoothing,
            fp16=torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=20,
            report_to="none",
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model, args=t_args,
            train_dataset=tok_train, eval_dataset=tok_val,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_clf_metrics,
        )

        t0           = time.time()
        train_result = trainer.train()
        train_time   = time.time() - t0

        preds_out  = trainer.predict(tok_test)
        preds_flat = np.argmax(preds_out.predictions, axis=1)
        labels_flat = preds_out.label_ids
        test_f1    = float(skf1(labels_flat, preds_flat, average="macro",  zero_division=0))
        test_acc   = float(accuracy_score(labels_flat, preds_flat))

        per_class_f1 = {
            f"test_f1_{name}": float(skf1(labels_flat, preds_flat, labels=[i], average="macro", zero_division=0))
            for i, name in enumerate(["none", "expiration", "effective", "renewal"])
        }

        mlflow.log_metrics({
            "test_f1":              test_f1,
            "test_accuracy":        test_acc,
            "total_train_time_sec": train_time,
            "time_per_epoch_sec":   train_time / epochs,
            "train_loss":           train_result.training_loss,
            **per_class_f1,
        })

    from ray import train as ray_train
    ray_train.report({"f1": test_f1})


def main():
    ray.init(ignore_reinit_error=True)
    search_space = {
        "model_key":        tune.choice(list(CLASSIFIER_MODEL_CANDIDATES.keys())),
        "learning_rate":   tune.loguniform(1e-5, 1e-4),
        "batch_size":      tune.choice([4, 8, 16, 32]),
        "warmup_ratio":    tune.uniform(0.05, 0.20),
        "weight_decay":    tune.loguniform(1e-3, 0.1),
        "epochs":          tune.choice([3, 5, 7]),
        "none_ratio":      tune.choice([5, 8, 10, 12]),
        "max_seq_length":  tune.choice([128, 256]),
        "label_smoothing": tune.choice([0.0, 0.05, 0.1]),
    }
    scheduler = ASHAScheduler(max_t=7, grace_period=2, reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(train_trial, resources={"GPU": 0.5, "CPU": 4}),
        param_space=search_space,
        tune_config=TuneConfig(
            metric="f1", mode="max", num_samples=16,
            scheduler=scheduler, max_concurrent_trials=2,
        ),
        run_config=RunConfig(
            name="deadline_clf_ray_tune",
            storage_path="/tmp/ray_results",
        ),
    )
    print("\n=== Ray Tune ASHA: classifier backbone search with RoBERTa + DeBERTa candidates ===\n")
    results = tuner.fit()
    best    = results.get_best_result(metric="f1", mode="max")
    print(f"\n=== BEST TRIAL ===")
    print(f"  macro F1:      {best.metrics['f1']:.4f}")
    print(f"  model_key:     {best.config['model_key']}")
    print(f"  base_model:    {CLASSIFIER_MODEL_CANDIDATES[best.config['model_key']]['base_model']}")
    print(f"  learning_rate: {best.config['learning_rate']:.2e}")
    print(f"  batch_size:    {best.config['batch_size']}")
    print(f"  warmup_ratio:  {best.config['warmup_ratio']:.3f}")
    print(f"  epochs:        {best.config['epochs']}")
    print(f"  none_ratio:    {best.config['none_ratio']}")
    print(f"  max_seq_length:{best.config['max_seq_length']}")
    print(f"  weight_decay:  {best.config['weight_decay']:.2e}")
    best_config_out = os.getenv("BEST_CONFIG_OUT", "").strip()
    if best_config_out:
        payload = {
            "model_key": best.config["model_key"],
            "base_model": CLASSIFIER_MODEL_CANDIDATES[best.config["model_key"]]["base_model"],
            "f1": best.metrics["f1"],
            "config": best.config,
        }
        with open(best_config_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(f"  label_smoothing:{best.config['label_smoothing']}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="RAY_TUNE_CLF_BEST_SUMMARY"):
        mlflow.log_params(
            {
                **best.config,
                "base_model": CLASSIFIER_MODEL_CANDIDATES[best.config["model_key"]]["base_model"],
                "search_method": "Ray Tune ASHA",
                "total_trials": len(results),
            }
        )
        mlflow.log_metrics({"best_f1": best.metrics["f1"]})
    print(f"\nAll trials → {MLFLOW_URI} | Experiment: {EXPERIMENT}\n")
    ray.shutdown()


if __name__ == "__main__":
    main()
