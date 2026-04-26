# Q2.2 — Live Object Storage Bucket

## Containers Created
- `cuad-data-proj11-v2` — v2 container (510-row dataset, converted from parquet)

## Files in cuad-data-proj11-v2
- data/train.jsonl (373 rows)
- data/validation.jsonl (68 rows)
- data/test.jsonl (69 rows)
- data/cuad_cleaned.jsonl (510 rows)
- raw/master_clauses.csv (510 rows)

## Public Bucket URL
https://chi.tacc.chameleoncloud.org/project/containers/container/cuad-data-proj11-v2

## Dataset Source
HuggingFace: tanvitakavane/datanauts_project_cuad-deadline-ner-version2

## Bootstrap Artifact Layout

For fresh-instance bootstrap, publish these additional objects:

- `bootstrap/data/train.jsonl`
- `bootstrap/data/test.jsonl`
- `bootstrap/models/deadline-ner-bert_ner_v1.tar.gz`
- `bootstrap/models/deadline-clf-roberta_clf_v6.tar.gz`
- `bootstrap/models/onnx_quantized_model.tar.gz`

Use:

```bash
bash scripts/publish-bootstrap-artifacts.sh
```

Then a new node can run:

```bash
bash scripts/bootstrap-production.sh <PUBLIC_IP>
```

and the bootstrap will download baseline data and model artifacts from object storage whenever they are not already present locally.
