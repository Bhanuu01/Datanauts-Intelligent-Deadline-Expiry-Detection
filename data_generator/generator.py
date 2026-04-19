#!/usr/bin/env python3
import random, time, json, requests, logging, os
from faker import Faker
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
fake = Faker()

SERVICE_URL = os.getenv('SERVICE_URL', 'http://feature-service:8000')
RATE = int(os.getenv('REQUESTS_PER_MINUTE', '10'))
SLEEP = 60.0 / RATE

DOC_TYPES   = ['invoice','contract','insurance','tax_form','lease']
EVENT_TYPES = ['expiration','payment_due','renewal','effective']
TEMPLATES_INV = [
    'Payment of ${a} is due by {d}.',
    'Invoice #{i} — Amount: ${a}. Due: {d}.',
    'Please remit ${a} no later than {d}.',
    'Balance due ${a} on or before {d}.',
]
TEMPLATES_CON = [
    'This agreement shall terminate on {d}.',
    'Contract expires {d} unless renewed.',
    'Term ends {d}. Renewal requires 30 days notice.',
]

def make_date():
    return (datetime.now() + timedelta(days=random.randint(7,365))).strftime('%B %d, %Y')

def gen_sentence(doc_type):
    d = make_date()
    a = f'{random.randint(500,50000):,}.{random.randint(0,99):02d}'
    i = fake.bothify('INV-#####')
    if doc_type == 'invoice':
        return random.choice(TEMPLATES_INV).format(d=d, a=a, i=i)
    return random.choice(TEMPLATES_CON).format(d=d)

def gen_upload():
    dt = random.choice(DOC_TYPES)
    return {
        'event': 'upload',
        'document_id': fake.uuid4(),
        'filename': f'{fake.company().replace(" ","_")}_{dt}.pdf',
        'document_type': dt,
        'sentence': gen_sentence(dt),
        'document_metadata': {
            'document_type': dt,
            'section_header': random.choice(['Payment Terms','Term and Termination','Renewal']),
            'upload_timestamp': datetime.utcnow().isoformat()+'Z'
        },
        'timestamp': datetime.utcnow().isoformat()
    }

def gen_feedback(doc_id):
    action = random.choice(['confirm','dismiss','edit','manual_add'])
    return {
        'event': action,
        'document_id': doc_id,
        'event_type': random.choice(EVENT_TYPES),
        'confidence': round(random.uniform(0.5,0.99),2),
        'timestamp': datetime.utcnow().isoformat()
    }

def send(payload, endpoint):
    try:
        r = requests.post(f'{SERVICE_URL}/{endpoint}', json=payload, timeout=5)
        log.info(f'POST /{endpoint} → HTTP {r.status_code} | doc={payload.get("document_id","")[:8]}')
    except requests.exceptions.ConnectionError:
        log.info(f'[SIMULATED] POST /{endpoint} | {json.dumps(payload)[:100]}')

def main():
    log.info(f'Generator started → {SERVICE_URL} at {RATE} req/min')
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
