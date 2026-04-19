#!/usr/bin/env python3
import random, time, json, requests, logging, os
from faker import Faker
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
fake = Faker()

SERVICE_URL = os.getenv('SERVICE_URL', 'http://feature-service:8000')
RATE = int(os.getenv('REQUESTS_PER_MINUTE', '10'))
SLEEP = 60.0 / RATE

DOC_TYPES = ['contract','lease','insurance','service_agreement']
EVENT_TYPES = ['expiration','effective','renewal','none']
TEMPLATES = [
    'This Agreement shall expire on {d}.',
    'Contract terminates on {d} unless renewed.',
    'The initial term ends {d} after Effective Date.',
    'Agreement effective as of {d}.',
    'Renewal notice required 30 days before {d}.',
]

def make_date():
    return (datetime.now() + timedelta(days=random.randint(30,730))).strftime('%B %d, %Y')

def gen_upload():
    dt = random.choice(DOC_TYPES)
    d = make_date()
    sentence = random.choice(TEMPLATES).format(d=d)
    return {
        'event': 'upload',
        'document_id': fake.uuid4(),
        'filename': f'{fake.company().replace(" ","_")}_{d}.pdf',
        'document_type': dt,
        'sentence': sentence,
        'document_metadata': {
            'document_type': dt,
            'section_header': random.choice(['Term','Renewal','Effective Date']),
            'upload_timestamp': datetime.utcnow().isoformat()+'Z'
        },
        'timestamp': datetime.utcnow().isoformat()
    }

def gen_feedback(doc_id):
    return {
        'event': random.choice(['confirm','dismiss','edit','manual_add']),
        'document_id': doc_id,
        'event_type': random.choice(EVENT_TYPES),
        'confidence': round(random.uniform(0.5,0.99),2),
        'timestamp': datetime.utcnow().isoformat()
    }

def send(payload, endpoint):
    try:
        r = requests.post(f'{SERVICE_URL}/{endpoint}', json=payload, timeout=5)
        log.info(f'POST /{endpoint} HTTP {r.status_code} doc={payload.get("document_id","")[:8]}')
    except:
        log.info(f'[SIMULATED] POST /{endpoint} | {json.dumps(payload)[:100]}')

def main():
    log.info(f'Generator started -> {SERVICE_URL} at {RATE} req/min')
    doc_ids, count = [], 0
    while True:
        count += 1
        if random.random() < 0.6 or not doc_ids:
            p = gen_upload()
            send(p, 'ingest')
            doc_ids.append(p['document_id'])
            if len(doc_ids) > 100: doc_ids.pop(0)
        else:
            send(gen_feedback(random.choice(doc_ids)), 'feedback')
        if count % 10 == 0:
            log.info(f'--- {count} events sent, {len(doc_ids)} docs in pool ---')
        time.sleep(SLEEP)

if __name__ == '__main__': main()
