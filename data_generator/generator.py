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

# Updated to match CLF_L2I: none, expiration, effective, renewal, agreement, notice_period
EVENT_TYPES = ['none', 'expiration', 'effective', 'renewal', 'agreement', 'notice_period']

TEMPLATES = {
    'expiration': [
        'This Agreement shall expire on {d}, unless terminated earlier by either party.',
        'The contract terminates on {d} unless renewed by written agreement.',
        'The software license granted hereunder shall terminate on {d} unless renewed.',
    ],
    'effective': [
        'This Agreement is effective as of {d} upon execution by both parties.',
        'The tenancy shall commence on {d} and the tenant agrees to all terms herein.',
        'Agreement effective as of {d}.',
    ],
    'renewal': [
        'The Agreement shall automatically renew for successive one-year terms unless either party provides written notice.',
        'This policy automatically renews annually on the anniversary date unless the insured provides 60 days written notice.',
        'Renewal notice required 30 days before {d}.',
    ],
    'agreement': [
        'This Agreement is entered into as of {d}, by and between the parties listed herein.',
        'This offer of employment is made as of {d} and is contingent upon successful background check completion.',
        'The Agreement is dated {d} and shall govern the relationship between the parties.',
    ],
    'notice_period': [
        'Either party may terminate this Agreement upon sixty (60) days prior written notice to the other party.',
        'To prevent automatic renewal, written cancellation notice must be received no later than 30 days prior to the renewal date.',
        'Written notice of termination must be delivered no later than {d}.',
    ],
    'none': [
        'Each party agrees to keep confidential all proprietary information disclosed under this Agreement.',
        'All goods must be delivered in accordance with the specifications outlined in Exhibit A.',
        'The employee agrees to maintain confidentiality of all proprietary information acquired during employment.',
    ],
}

def make_date():
    return (datetime.now() + timedelta(days=random.randint(30,730))).strftime('%B %d, %Y')

def gen_upload():
    dt = random.choice(DOC_TYPES)
    event_type = random.choice(EVENT_TYPES)
    d = make_date()
    template = random.choice(TEMPLATES[event_type])
    sentence = template.format(d=d) if '{d}' in template else template
    return {
        'event': 'upload',
        'document_id': fake.uuid4(),
        'filename': f'{fake.company().replace(" ","_")}_{d}.pdf',
        'document_type': dt,
        'sentence': sentence,
        'event_type': event_type,
        'document_metadata': {
            'document_type': dt,
            'section_header': random.choice(['Term','Renewal','Effective Date','Agreement Date','Notice']),
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
