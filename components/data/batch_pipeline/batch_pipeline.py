#!/usr/bin/env python3
"""
Batch pipeline: compiles versioned training and evaluation datasets.
- Candidate selection: well-justified eligibility filters
- Leakage prevention: time-based split anchored at agreement date
- Versioning: every run tagged with timestamp + source hash
- Dataset: tanvitakavane/datanauts_project_cuad-deadline-ner-version2
"""
import json, re, os, hashlib, subprocess, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BUCKET = os.environ.get('BUCKET_NAME', 'cuad-data-proj11-v2')
VERSION = datetime.utcnow().strftime('v%Y%m%d_%H%M%S')

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_year(s):
    # v2 dataset uses 'agreement_date_iso' field e.g. '1990-12-01'
    raw = str(s.get('agreement_date_iso', '') or '')
    m4 = re.search(r'(\d{4})', raw)
    if m4:
        yr = int(m4.group(1))
        if 1990 <= yr <= 2030:
            return yr
    # fallback to contract_date
    raw2 = str(s.get('contract_date', '') or '')
    m4b = re.search(r'(\d{4})', raw2)
    if m4b:
        yr2 = int(m4b.group(1))
        if 1990 <= yr2 <= 2030:
            return yr2
    return None

def candidate_selection(samples):
    kept, dropped = [], {'short': 0, 'no_ocr': 0}
    for s in samples:
        # v2 dataset uses 'ocr_text' and 'Filename'
        ocr = s.get('ocr_text', '') or ''
        if not ocr or len(ocr) < 15:
            dropped['short'] += 1
            continue
        if not s.get('Filename'):
            dropped['no_ocr'] += 1
            continue
        kept.append(s)
    log.info(f'Candidate selection: kept={len(kept)} dropped={dropped}')
    return kept

def time_based_split(samples):
    train, val, test = [], [], []
    assignments = {}
    for s in samples:
        fname = s.get('Filename', '')
        if fname in assignments:
            split = assignments[fname]
        else:
            yr = get_year(s)
            if yr is None or yr < 2005:
                split = 'train'
            elif yr <= 2010:
                split = 'val'
            else:
                split = 'test'
            assignments[fname] = split
        if split == 'train':
            train.append(s)
        elif split == 'val':
            val.append(s)
        else:
            test.append(s)
    return train, val, test

def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for s in data:
            f.write(json.dumps(s) + '\n')
    log.info(f'Saved {len(data)} -> {path}')

def upload(local, remote):
    r = subprocess.run(
        ['openstack', 'object', 'create', BUCKET, '--name', remote, local],
        capture_output=True, text=True)
    log.info(f'Uploaded {remote}' if r.returncode == 0 else f'FAILED: {r.stderr}')

def main():
    log.info(f'=== Batch Pipeline Run: {VERSION} ===')
    cuad = load_jsonl('/data/cuad_cleaned.jsonl')
    log.info(f'Loaded: {len(cuad)} CUAD records (v2 dataset — no synthetic)')

    source_hash = hashlib.sha256(
        open('/data/cuad_cleaned.jsonl', 'rb').read()
    ).hexdigest()[:12]

    eligible = candidate_selection(cuad)
    train, val, test = time_based_split(eligible)

    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        for s in split_data:
            s['_version'] = VERSION
            s['_split'] = split_name
            s['_source_hash'] = source_hash

    out = f'/output/{VERSION}'
    save_jsonl(train, f'{out}/train.jsonl')
    save_jsonl(val,   f'{out}/validation.jsonl')
    save_jsonl(test,  f'{out}/test.jsonl')

    manifest = {
        'version': VERSION,
        'pipeline_run': datetime.utcnow().isoformat(),
        'source_hash': source_hash,
        'dataset': 'tanvitakavane/datanauts_project_cuad-deadline-ner-version2',
        'synthetic_data': False,
        'counts': {'train': len(train), 'val': len(val), 'test': len(test)},
        'split_logic': 'time-based: pre-2005=train, 2005-2010=val, 2011+=test',
        'leakage_prevention': 'contract-level grouping — all clauses of one contract in same split',
        'frozen_test': True,
        'candidate_selection': 'ocr_text>=15chars, has Filename'
    }
    mpath = f'{out}/manifest.json'
    with open(mpath, 'w') as f:
        json.dump(manifest, f, indent=2)

    for local, remote in [
        (f'{out}/train.jsonl',      f'versioned/{VERSION}/train.jsonl'),
        (f'{out}/validation.jsonl', f'versioned/{VERSION}/validation.jsonl'),
        (f'{out}/test.jsonl',       f'versioned/{VERSION}/test.jsonl'),
        (f'{out}/manifest.json',    f'versioned/{VERSION}/manifest.json'),
        (f'{out}/train.jsonl',      'latest/train.jsonl'),
        (f'{out}/validation.jsonl', 'latest/validation.jsonl'),
        (f'{out}/test.jsonl',       'latest/test.jsonl'),
        (mpath,                     'latest/manifest.json'),
    ]:
        upload(local, remote)

    log.info(f'=== COMPLETE: {VERSION} ===')
    r = subprocess.run(
        ['openstack', 'object', 'list', BUCKET, '--prefix', f'versioned/{VERSION}'],
        capture_output=True, text=True)
    log.info('External confirmation (bucket contents):')
    log.info(r.stdout)

if __name__ == '__main__':
    main()
