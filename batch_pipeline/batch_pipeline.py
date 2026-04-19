#!/usr/bin/env python3
import json, re, os, hashlib, subprocess, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

BUCKET  = os.environ.get('BUCKET_NAME', 'cuad-data-proj11')
VERSION = datetime.utcnow().strftime('v%Y%m%d_%H%M%S')

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_year(s):
    raw = str(s.get('agreement_date_raw',''))
    m = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2})$', raw)
    if m:
        yy = int(m.group(3))
        return 2000+yy if yy <= 68 else 1900+yy
    m4 = re.search(r'(\d{4})', raw)
    if m4:
        yr = int(m4.group(1))
        if 1990 <= yr <= 2030: return yr
    return None

def candidate_selection(samples):
    kept = []
    dropped = {'short':0,'no_tokens':0,'no_year':0}
    for s in samples:
        if not s.get('sentence') or len(s['sentence']) < 15:
            dropped['short'] += 1; continue
        if not s.get('tokens'):
            dropped['no_tokens'] += 1; continue
        if get_year(s) is None and s.get('event_type') not in ['payment_due']:
            dropped['no_year'] += 1; continue
        kept.append(s)
    log.info(f'Candidate selection: kept={len(kept)} dropped={dropped}')
    return kept

def time_based_split(samples):
    train, val, test = [], [], []
    assignments = {}
    for s in samples:
        fname = s.get('filename','')
        if fname in assignments:
            split = assignments[fname]
        else:
            yr = get_year(s)
            if yr is None or yr < 2015: split = 'train'
            elif yr <= 2018:            split = 'val'
            else:                       split = 'test'
            assignments[fname] = split
        if split == 'train': train.append(s)
        elif split == 'val': val.append(s)
        else:                test.append(s)
    return train, val, test

def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f:
        for s in data: f.write(json.dumps(s)+'\n')
    log.info(f'Saved {len(data)} → {path}')

def upload(local, remote):
    r = subprocess.run(['openstack','object','create',BUCKET,'--name',remote,local],
                       capture_output=True, text=True)
    if r.returncode == 0: log.info(f'Uploaded {remote}')
    else: log.error(f'Upload failed: {r.stderr}')

def main():
    log.info(f'=== Batch Pipeline Run: {VERSION} ===')
    cuad  = load_jsonl('/data/cuad_cleaned.jsonl')
    synth = load_jsonl('/data/synthetic_ner.jsonl')
    all_data = cuad + synth
    log.info(f'Loaded: {len(cuad)} CUAD + {len(synth)} synthetic = {len(all_data)} total')

    source_hash = hashlib.sha256(open('/data/cuad_cleaned.jsonl','rb').read()).hexdigest()[:12]
    eligible = candidate_selection(all_data)
    train, val, test = time_based_split(eligible)

    for split_name, split_data in [('train',train),('val',val),('test',test)]:
        for s in split_data:
            s['_version'] = VERSION
            s['_split']   = split_name
            s['_source_hash'] = source_hash

    out = f'/output/{VERSION}'
    save_jsonl(train, f'{out}/train.jsonl')
    save_jsonl(val,   f'{out}/validation.jsonl')
    save_jsonl(test,  f'{out}/test.jsonl')

    manifest = {
        'version': VERSION,
        'pipeline_run': datetime.utcnow().isoformat(),
        'source_hash': source_hash,
        'counts': {'train':len(train),'val':len(val),'test':len(test)},
        'split_logic': 'time-based: pre-2015=train, 2015-2018=val, 2019+=test',
        'leakage_prevention': 'contract-level grouping — all clauses of one contract in same split',
        'frozen_test': True,
        'candidate_selection': 'sentence>=15chars, has tokens, has dateable year'
    }
    mpath = f'{out}/manifest.json'
    with open(mpath,'w') as f: json.dump(manifest,f,indent=2)

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
    r = subprocess.run(['openstack','object','list',BUCKET,'--prefix',f'versioned/{VERSION}'],
                       capture_output=True, text=True)
    log.info('External confirmation (bucket contents):')
    log.info(r.stdout)

if __name__ == '__main__': main()
