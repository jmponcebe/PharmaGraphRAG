from fastapi.testclient import TestClient
from pharmagraphrag.api.main import app
client = TestClient(app)
resp = client.post('/query', json={'question': 'What are the side effects of aspirin?', 'use_llm': True})
print(f'Status: {resp.status_code}')
data = resp.json()
print(f'Answer preview: {data.get("answer", "")[:200]}')
print(f'Drugs found: {data.get("drugs_found_in_graph", [])}')
print(f'Error: {data.get("error")}')
from pharmagraphrag.observability import flush
flush()
print('Langfuse flushed')
