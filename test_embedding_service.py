import pytest
from fastapi.testclient import TestClient
from embed_endpoint import app, Settings, EmbeddingService
from sentence_transformers import SentenceTransformer

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_text():
    with open("tests/test_data/sample_opinion.txt") as f:
        return f.read()

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "gpu_available" in data

def test_heartbeat(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == '"Heartbeat detected."'

def test_query_embedding(client):
    response = client.post(
        "/api/v1/embed/query",
        json={"text": "What constitutes copyright infringement?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert isinstance(data["embedding"], list)

def test_text_embedding(client, sample_text):
    response = client.post(
        "/api/v1/embed/text",
        data=sample_text,
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert isinstance(data["embeddings"], list)

def test_query_embedding_error(client):
    response = client.post(
        "/api/v1/embed/query",
        json={"text": ""}
    )
    assert response.status_code == 500

def test_text_embedding_error(client):
    response = client.post(
        "/api/v1/embed/text",
        data="",
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 500

def test_service_unavailable(client, monkeypatch):
    monkeypatch.setattr("embed_endpoint.embedding_service", None)
    response = client.post(
        "/api/v1/embed/query",
        json={"text": "test"}
    )
    assert response.status_code == 503
