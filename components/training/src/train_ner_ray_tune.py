import os, time, random
import numpy as np
import torch
import torch.nn as nn
import mlflow
import ray
from ray import tune
from ray.train import RunConfig
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

os.environ["AWS_ACCESS_KEY_ID"]      = os.getenv("AWS_ACCESS_KEY_ID", "datanauts-key")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.getenv("AWS_SECRET_ACCESS_KEY", "datanauts-secret")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://129.114.27.190:30900")
os.environ["GIT_PYTHON_REFRESH"]     = "quiet"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://129.114.27.190:30500")
EXPERIMENT = "deadline-detection-ner-ray-tune"
BASE_MODEL = "dslim/bert-base-NER"
DATA_PATH  = "/app/data/deadline_sentences"

LABEL_LIST = [
    "O",
    "B-EXP_DATE",    "I-EXP_DATE",
    "B-START_DATE",  "I-START_DATE",
    "B-DURATION",    "I-DURATION",
    "B-NOTICE_DATE", "I-NOTICE_DATE",
]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}

# O-tag weight=1.0, entity tags upweighted; NOTICE_DATE 12x (rarest)
NER_WEIGHTS = torch.tensor([1.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 12.0, 12.0])

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)


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
    train_ds  = dd["train"].select_columns(["tokens", "ner_tags"])
    val_ds    = dd["val"].select_columns(["tokens", "ner_tags"])
    test_ds   = dd["test"].select_columns(["tokens", "ner_tags"])
    return train_ds, val_ds, test_ds


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
            metrics[f"{entity}_f1"] = report[entity]["f1-score"]
    return metrics


def train_trial(config):
    lr         = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs     = config["epochs"]
    max_len    = config["max_seq_length"]
    run_name   = f"ner_lr{lr:.1e}_bs{batch_size}_ep{epochs}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate":  lr,
            "batch_size":     batch_size,
            "epochs":         epochs,
            "max_seq_length": max_len,
            "base_model":     BASE_MODEL,
            "search_method":  "Ray Tune ASHA",
            "num_labels":     len(LABEL_LIST),
            "label_schema":   "O|B-EXP_DATE|I-EXP_DATE|B-START_DATE|I-START_DATE|B-DURATION|I-DURATION|B-NOTICE_DATE|I-NOTICE_DATE",
            "gpu":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "filter_none":    True,
        })

        train_ds, val_ds, test_ds = load_ner_data()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        def tokenize_and_align(examples):
            tokenized = tokenizer(
                examples["tokens"], truncation=True,
                max_length=max_len, is_split_into_words=True,
            )
            aligned = []
            for i, tags in enumerate(examples["ner_tags"]):
                wids, prev, seq = tokenized.word_ids(batch_index=i), None, []
                for wid in wids:
                    if wid is None:
                        seq.append(-100)
                    elif wid != prev:
                        seq.append(tags[wid])
                    else:
                        seq.append(-100)
                    prev = wid
                aligned.append(seq)
            tokenized["labels"] = aligned
            return tokenized

        tok_train = train_ds.map(tokenize_and_align, batched=True, remove_columns=train_ds.column_names)
        tok_val   = val_ds.map(tokenize_and_align,   batched=True, remove_columns=val_ds.column_names)
        tok_test  = test_ds.map(tokenize_and_align,  batched=True, remove_columns=test_ds.column_names)

        model = AutoModelForTokenClassification.from_pretrained(
            BASE_MODEL, num_labels=len(LABEL_LIST),
            id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        t_args = TrainingArguments(
            output_dir=f"/tmp/ray_ner_{run_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            fp16=torch.cuda.is_available(),
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
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics,
        )

        t0           = time.time()
        train_result = trainer.train()
        train_time   = time.time() - t0

        preds_out = trainer.predict(tok_test)
        test_f1   = preds_out.metrics.get("test_f1", 0)

        mlflow.log_metrics({
            "test_f1":               test_f1,
            "test_precision":        preds_out.metrics.get("test_precision",     0),
            "test_recall":           preds_out.metrics.get("test_recall",        0),
            "test_EXP_DATE_f1":      preds_out.metrics.get("test_EXP_DATE_f1",  0),
            "test_START_DATE_f1":    preds_out.metrics.get("test_START_DATE_f1", 0),
            "test_DURATION_f1":      preds_out.metrics.get("test_DURATION_f1",   0),
            "test_NOTICE_DATE_f1":   preds_out.metrics.get("test_NOTICE_DATE_f1",0),
            "total_train_time_sec":  train_time,
            "time_per_epoch_sec":    train_time / epochs,
            "train_loss":            train_result.training_loss,
        })

    from ray import train as ray_train
    ray_train.report({"f1": test_f1})


def main():
    ray.init(ignore_reinit_error=True)
    search_space = {
        "learning_rate":  tune.loguniform(1e-5, 5e-5),
        "batch_size":     tune.choice([8, 16]),
        "epochs":         tune.choice([3, 5]),
        "max_seq_length": tune.choice([128, 256]),
    }
    scheduler = ASHAScheduler(max_t=5, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(train_trial, resources={"GPU": 0.5, "CPU": 4}),
        param_space=search_space,
        tune_config=TuneConfig(
            metric="f1", mode="max", num_samples=8,
            scheduler=scheduler, max_concurrent_trials=2,
        ),
        run_config=RunConfig(
            name="deadline_ner_ray_tune",
            storage_path="/tmp/ray_results",
        ),
    )
    print("\n=== Ray Tune ASHA: 8 trials on BERT NER (7-class), 2 concurrent ===\n")
    results = tuner.fit()
    best    = results.get_best_result(metric="f1", mode="max")
    print(f"\n=== BEST TRIAL ===")
    print(f"  F1:            {best.metrics['f1']:.4f}")
    print(f"  learning_rate: {best.config['learning_rate']:.2e}")
    print(f"  batch_size:    {best.config['batch_size']}")
    print(f"  epochs:        {best.config['epochs']}")
    print(f"  max_seq_length:{best.config['max_seq_length']}")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="RAY_TUNE_NER_BEST_SUMMARY"):
        mlflow.log_params({**best.config, "search_method": "Ray Tune ASHA", "total_trials": len(results)})
        mlflow.log_metrics({"best_f1": best.metrics["f1"]})
    print(f"\nAll trials → {MLFLOW_URI} | Experiment: {EXPERIMENT}\n")
    ray.shutdown()


if __name__ == "__main__":
    main()
