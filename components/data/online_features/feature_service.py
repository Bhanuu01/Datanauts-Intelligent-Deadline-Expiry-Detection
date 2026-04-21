import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import redis
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from pydantic import BaseModel
from starlette.responses import Response

# Use simple sentence splitting instead of NLTK to avoid download issues
app = FastAPI(title='Deadline Detection Online Feature Service')

INGEST_REQUESTS = Counter(
    'online_features_ingest_requests_total',
    'Total number of ingest requests processed by the online feature service.',
)
INGEST_CANDIDATES = Counter(
    'online_features_candidate_sentences_total',
    'Total number of candidate sentences emitted by the online feature service.',
)
REDIS_ERRORS = Counter(
    'online_features_redis_errors_total',
    'Total number of Redis interaction failures in the online feature service.',
)
INGEST_LATENCY = Histogram(
    'online_features_ingest_latency_seconds',
    'Latency of ingest requests handled by the online feature service.',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

r = redis.Redis(
    host=os.getenv('REDIS_HOST','redis'),
    port=int(os.getenv('REDIS_PORT','6379')),
    decode_responses=True)

DATE_RE = re.compile(
    r'\b(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}'
    r'|\b\d{1,2}/\d{1,2}/\d{2,4}'
    r'|\b\d{4}-\d{2}-\d{2}', re.IGNORECASE)

SECTIONS = {
    'payment':'Payment Terms','expir':'Term and Termination',
    'terminat':'Term and Termination','renew':'Renewal',
    'effective':'Effective Date','due':'Payment Terms'
}

class IngestRequest(BaseModel):
    document_id: str = None
    ocr_text: str
    document_type: str = 'unknown'
    filename: str = 'document.pdf'


class FeedbackRequest(BaseModel):
    event: str
    document_id: str
    event_type: str = 'none'
    confidence: float = 0.0
    timestamp: str | None = None
    notes: str | None = None


def append_jsonl(path_str: str, payload: dict):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload) + '\n')

def detect_section(s):
    for k, v in SECTIONS.items():
        if k in s.lower(): return v
    return 'Unknown'

def simple_sent_tokenize(text):
    # Simple sentence splitter without NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

@app.post('/ingest')
def ingest(req: IngestRequest):
    INGEST_REQUESTS.inc()
    with INGEST_LATENCY.time():
        doc_id = req.document_id or str(uuid.uuid4())
        sents = simple_sent_tokenize(req.ocr_text)
        features = []
        for i, sent in enumerate(sents):
            sent = sent.strip()
            if len(sent) < 10:
                continue
            has_date = bool(DATE_RE.search(sent))
            features.append({
                'sentence': sent, 'tokens': sent.split(),
                'context_window': {
                    'prev_sentence': sents[i-1].strip() if i > 0 else '',
                    'next_sentence': sents[i+1].strip() if i < len(sents) - 1 else ''},
                'document_metadata': {
                    'document_type': req.document_type,
                    'section_header': detect_section(sent),
                    'upload_timestamp': datetime.utcnow().isoformat() + 'Z',
                    'filename': req.filename},
                'has_date_candidate': has_date,
                'sentence_index': i, 'doc_id': doc_id
            })
        try:
            r.setex(f'features:{doc_id}', 3600, json.dumps(features))
        except Exception:
            REDIS_ERRORS.inc()
            raise

        candidates = [f for f in features if f['has_date_candidate']]
        INGEST_CANDIDATES.inc(len(candidates))
        append_jsonl(
            os.getenv('PRODUCTION_DATA_LOG_PATH', '/data/production_ingest.jsonl'),
            {
                'event': 'upload',
                'document_id': doc_id,
                'document_type': req.document_type,
                'filename': req.filename,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'sentence_count': len(features),
                'candidate_count': len(candidates),
                'features': candidates,
            },
        )
        return {
            'document_id': doc_id, 'sentences': len(features),
            'candidates': len(candidates), 'features': candidates,
            'redis_key': f'features:{doc_id}'
        }

@app.get('/features/{doc_id}')
def get_features(doc_id: str):
    data = r.get(f'features:{doc_id}')
    return {'doc_id': doc_id, 'features': json.loads(data)} if data else {'error': 'not found'}

@app.get('/health')
def health():
    try:
        r.ping()
        ok = True
    except Exception:
        ok = False
        REDIS_ERRORS.inc()
    return {'status': 'ok', 'redis': ok}


@app.post('/feedback')
def feedback(req: FeedbackRequest):
    payload = req.model_dump()
    payload['timestamp'] = payload['timestamp'] or datetime.utcnow().isoformat() + 'Z'
    append_jsonl(os.getenv('FEEDBACK_LOG_PATH', '/data/feedback_events.jsonl'), payload)
    return {'status': 'recorded', 'document_id': req.document_id}


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
