import subprocess, os, sys

BUCKET = os.environ.get('BUCKET_NAME', 'cuad-data-proj11-v2')
FILES = [
    ('output/raw/master_clauses.csv',    'raw/master_clauses.csv'),
    ('output/data/cuad_cleaned.jsonl',   'data/cuad_cleaned.jsonl'),
    ('output/data/train.jsonl',          'data/train.jsonl'),
    ('output/data/validation.jsonl',     'data/validation.jsonl'),
    ('output/data/test.jsonl',           'data/test.jsonl'),
]

subprocess.run(['openstack', 'container', 'create', BUCKET], check=False)
subprocess.run(['openstack', 'container', 'set',
    '--property', 'X-Container-Read=.r:*,.rlistings', BUCKET], check=True)

for local_path, object_name in FILES:
    if not os.path.exists(local_path):
        print(f'MISSING: {local_path}'); sys.exit(1)
    r = subprocess.run(
        ['openstack', 'object', 'create', '--name', object_name, BUCKET, local_path],
        capture_output=True, text=True)
    print(f'Uploaded: {object_name}' if r.returncode == 0 else f'FAILED: {r.stderr}')

result = subprocess.run(['openstack', 'object', 'list', BUCKET],
    capture_output=True, text=True)
print('=== BUCKET CONTENTS (external confirmation) ===')
print(result.stdout)
