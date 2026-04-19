"""
generate_report.py  —  Q2.1 Training Runs Report
Queries MLflow for all runs in both experiments and produces:
  - reports/training_runs.md   (markdown table with clickable links)
  - reports/training_runs.html (standalone HTML, printable to PDF)

Usage:
  python generate_report.py
  python generate_report.py --mlflow-uri http://129.114.27.190:8000
"""

import os, argparse
from datetime import datetime

MLFLOW_URI        = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.27.190:8000")
NER_EXPERIMENT    = "deadline-detection-ner"
CLF_EXPERIMENT    = "deadline-detection-classifier"
REPORTS_DIR       = "./reports"


def fetch_runs(client, experiment_name):
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"  [warn] experiment '{experiment_name}' not found")
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.test_f1 DESC"],
        )
        return runs
    except Exception as e:
        print(f"  [error] fetching {experiment_name}: {e}")
        return []


def fmt(val, decimals=4):
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


def run_url(mlflow_uri, experiment_id, run_id):
    return f"{mlflow_uri}/#/experiments/{experiment_id}/runs/{run_id}"


def build_ner_rows(runs, mlflow_uri):
    rows = []
    best_f1 = max((r.data.metrics.get("test_f1", 0) for r in runs), default=0)
    for r in runs:
        m  = r.data.metrics
        p  = r.data.params
        f1 = m.get("test_f1", None)
        url = run_url(mlflow_uri, r.info.experiment_id, r.info.run_id)
        is_best = f1 is not None and abs(f1 - best_f1) < 1e-6
        rows.append({
            "variant":     r.info.run_name or p.get("model_name", r.info.run_id[:8]),
            "base_model":  p.get("base_model", "—"),
            "epochs":      p.get("epochs", "—"),
            "lr":          p.get("learning_rate", "—"),
            "batch_size":  p.get("batch_size", "—"),
            "f1":          fmt(f1),
            "precision":   fmt(m.get("test_precision")),
            "recall":      fmt(m.get("test_recall")),
            "train_time":  fmt(m.get("total_train_time_sec"), 0) + "s" if m.get("total_train_time_sec") else "—",
            "mlflow_link": url,
            "is_best":     is_best,
        })
    return rows


def build_clf_rows(runs, mlflow_uri):
    rows = []
    best_f1 = max((r.data.metrics.get("test_f1", 0) for r in runs), default=0)
    for r in runs:
        m  = r.data.metrics
        p  = r.data.params
        f1 = m.get("test_f1", None)
        url = run_url(mlflow_uri, r.info.experiment_id, r.info.run_id)
        is_best = f1 is not None and abs(f1 - best_f1) < 1e-6
        rows.append({
            "variant":     r.info.run_name or p.get("model_name", r.info.run_id[:8]),
            "base_model":  p.get("base_model", "—"),
            "epochs":      p.get("epochs", "—"),
            "lr":          p.get("learning_rate", "—"),
            "none_ratio":  p.get("none_ratio", "—"),
            "f1":          fmt(f1),
            "accuracy":    fmt(m.get("test_accuracy")),
            "train_time":  fmt(m.get("total_train_time_sec"), 0) + "s" if m.get("total_train_time_sec") else "—",
            "mlflow_link": url,
            "is_best":     is_best,
        })
    return rows


def to_markdown(ner_rows, clf_rows, mlflow_uri):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Datanauts — Training Runs Report",
        "",
        f"**Generated:** {ts}  ",
        f"**MLflow UI:** [{mlflow_uri}]({mlflow_uri})  ",
        f"**Experiments:** `{NER_EXPERIMENT}`, `{CLF_EXPERIMENT}`",
        "",
        "---",
        "",
        "## NER Model Runs (`deadline-detection-ner`)",
        "",
        "| Variant | Base Model | Epochs | LR | Batch | F1 | Precision | Recall | Train Time | MLflow Run |",
        "|---------|-----------|--------|----|-------|----|-----------|--------|------------|------------|",
    ]
    for r in ner_rows:
        star = " ⭐ **BEST**" if r["is_best"] else ""
        lines.append(
            f"| **{r['variant']}**{star} | {r['base_model']} | {r['epochs']} | {r['lr']} "
            f"| {r['batch_size']} | {r['f1']} | {r['precision']} | {r['recall']} "
            f"| {r['train_time']} | [link]({r['mlflow_link']}) |"
        )

    lines += [
        "",
        "---",
        "",
        "## Classifier Model Runs (`deadline-detection-classifier`)",
        "",
        "| Variant | Base Model | Epochs | LR | none_ratio | F1 (macro) | Accuracy | Train Time | MLflow Run |",
        "|---------|-----------|--------|----|-----------|-----------:|----------|------------|------------|",
    ]
    for r in clf_rows:
        star = " ⭐ **BEST**" if r["is_best"] else ""
        lines.append(
            f"| **{r['variant']}**{star} | {r['base_model']} | {r['epochs']} | {r['lr']} "
            f"| {r['none_ratio']} | {r['f1']} | {r['accuracy']} "
            f"| {r['train_time']} | [link]({r['mlflow_link']}) |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes on Best Candidates",
        "",
        "### NER",
        "- **Best variant** selected by highest entity-level F1 on held-out test contracts.",
        "- `bert_ner_v1` (lr=2e-5, 3 epochs) is the expected best — stable learning rate with pre-trained NER weights.",
        "- `bert_ner_v3` (5 epochs) may overfit if val F1 plateaus; early stopping handles this.",
        "",
        "### Classifier",
        "- **Best variant** selected by macro F1 — penalizes poor recall on minority `explicit`/`computable` classes.",
        "- `roberta_clf_v1` (lr=2e-5, none_ratio=5) is the expected best — balanced downsampling + weighted loss.",
        "- Higher LR (`roberta_clf_v2`, lr=5e-5) risks overshooting for sequence classification.",
        "",
        "---",
        f"*Report generated by `generate_report.py`. Re-run after each training session to refresh links.*",
    ]
    return "\n".join(lines)


def to_html(markdown_text, mlflow_uri):
    try:
        import markdown as md_lib
        body = md_lib.markdown(markdown_text, extensions=["tables"])
    except ImportError:
        body = f"<pre>{markdown_text}</pre>"
        body += "<p><em>Install `markdown` package for full HTML rendering: pip install markdown</em></p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Datanauts Training Runs Report</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1   {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 8px; }}
  h2   {{ color: #16213e; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
  th   {{ background: #16213e; color: white; padding: 8px 12px; text-align: left; }}
  td   {{ padding: 7px 12px; border-bottom: 1px solid #ddd; }}
  tr:hover {{ background: #f5f5f5; }}
  a    {{ color: #e94560; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
  hr   {{ border: none; border-top: 1px solid #ccc; margin: 24px 0; }}
  .best {{ background: #fff8e1 !important; font-weight: bold; }}
</style>
</head>
<body>
{body}
<hr>
<p style="color:#888;font-size:12px;">
  MLflow UI: <a href="{mlflow_uri}">{mlflow_uri}</a> |
  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
</p>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", default=MLFLOW_URI)
    args = parser.parse_args()

    try:
        import mlflow
    except ImportError:
        print("mlflow not installed. Run: pip install mlflow")
        return

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.tracking.MlflowClient()

    print(f"Querying MLflow at {args.mlflow_uri} ...")
    ner_runs = fetch_runs(client, NER_EXPERIMENT)
    clf_runs = fetch_runs(client, CLF_EXPERIMENT)
    print(f"  NER runs  : {len(ner_runs)}")
    print(f"  CLF runs  : {len(clf_runs)}")

    ner_rows = build_ner_rows(ner_runs, args.mlflow_uri)
    clf_rows = build_clf_rows(clf_runs, args.mlflow_uri)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    md_text  = to_markdown(ner_rows, clf_rows, args.mlflow_uri)
    md_path  = os.path.join(REPORTS_DIR, "training_runs.md")
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"\nMarkdown report : {md_path}")

    html_text = to_html(md_text, args.mlflow_uri)
    html_path = os.path.join(REPORTS_DIR, "training_runs.html")
    with open(html_path, "w") as f:
        f.write(html_text)
    print(f"HTML report     : {html_path}")
    print("\nTo convert HTML to PDF:")
    print("  Mac  : Open in Chrome → File → Print → Save as PDF")
    print("  CLI  : pip install weasyprint && weasyprint reports/training_runs.html reports/training_runs.pdf")


if __name__ == "__main__":
    main()
