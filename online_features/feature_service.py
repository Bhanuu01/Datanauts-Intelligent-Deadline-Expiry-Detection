from fastapi import FastAPI
from pydantic import BaseModel
import redis, re, json, os, uuid
from datetime import datetime

# Use simple sentence splitting instead of NLTK to avoid download issues
app = FastAPI(title='Deadline Detection Online Feature Service')

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
    doc_id = req.document_id or str(uuid.uuid4())
    sents = simple_sent_tokenize(req.ocr_text)
    features = []
    for i, sent in enumerate(sents):
        sent = sent.strip()
        if len(sent) < 10: continue
        has_date = bool(DATE_RE.search(sent))
        features.append({
            'sentence': sent, 'tokens': sent.split(),
            'context_window': {
                'prev_sentence': sents[i-1].strip() if i>0 else '',
                'next_sentence': sents[i+1].strip() if i<len(sents)-1 else ''},
            'document_metadata': {
                'document_type': req.document_type,
                'section_header': detect_section(sent),
                'upload_timestamp': datetime.utcnow().isoformat()+'Z',
                'filename': req.filename},
            'has_date_candidate': has_date,
            'sentence_index': i, 'doc_id': doc_id
        })
    r.setex(f'features:{doc_id}', 3600, json.dumps(features))
    candidates = [f for f in features if f['has_date_candidate']]
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
    try: r.ping(); ok=True
    except: ok=False
    return {'status': 'ok', 'redis': ok}
