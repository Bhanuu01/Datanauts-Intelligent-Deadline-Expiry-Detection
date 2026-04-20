import requests, time, numpy as np
from concurrent.futures import ThreadPoolExecutor

payload = {"document_id": "123", "ocr_text": "Invoice due May 1 2026."}

def test_api(url):
    latencies = []
    start_total = time.time()
    
    def send_req():
        s = time.time()
        requests.post(url, json=payload)
        return time.time() - s

    with ThreadPoolExecutor(max_workers=10) as ex:
        latencies = list(ex.map(lambda _: send_req(), range(200)))
        
    throughput = 200 / (time.time() - start_total)
    print(f"URL: {url} | T-put: {throughput:.2f} req/s | p95: {np.percentile(latencies, 95)*1000:.2f}ms")

test_api("http://localhost:8000/predict") # Tests Baseline
test_api("http://localhost:8001/predict") # Tests ONNX
