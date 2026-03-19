import pytest
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)

class TestAPI:
    def test_health(self): assert client.get("/health").status_code == 200
    def test_chat(self):
        resp = client.post("/chat", json={"text": "hello"})
        assert resp.status_code == 200
        assert "response" in resp.json()
