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

def test_query_embedding_validation(client):
    # Test empty text
    response = client.post(
        "/api/v1/embed/query",
        json={"text": ""}
    )
    assert response.status_code == 422
    
    # Test text too long
    response = client.post(
        "/api/v1/embed/query",
        json={"text": "a" * (Settings().max_text_length + 1)}
    )
    assert response.status_code == 422

def test_text_embedding_validation(client):
    # Test empty text
    response = client.post(
        "/api/v1/embed/text",
        data="",
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 422
    
    # Test text too long
    response = client.post(
        "/api/v1/embed/text",
        data="a" * (Settings().max_text_length + 1),
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 422
    
    # Test invalid UTF-8
    response = client.post(
        "/api/v1/embed/text",
        data=bytes([0xFF, 0xFE, 0xFD]),
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 422

def test_service_unavailable(client, monkeypatch):
    monkeypatch.setattr("embed_endpoint.embedding_service", None)
    response = client.post(
        "/api/v1/embed/query",
        json={"text": "test"}
    )
    assert response.status_code == 503
    assert "service not initialized" in response.json()["detail"].lower()

def test_batch_processing(client):
    batch_request = {
        "documents": [
            {"id": 1, "text": "First test document"},
            {"id": 2, "text": "Second test document"}
        ]
    }
    response = client.post("/api/v1/embed/batch", json=batch_request)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert all(isinstance(doc["embeddings"], list) for doc in data)
    assert all(doc["id"] in [1, 2] for doc in data)

def test_batch_validation(client):
    # Test batch size limit
    settings = Settings()
    large_batch = {
        "documents": [
            {"id": i, "text": f"Document {i}"}
            for i in range(settings.max_batch_size + 1)
        ]
    }
    response = client.post("/api/v1/embed/batch", json=large_batch)
    assert response.status_code == 422
    assert "batch size exceeds maximum" in response.json()["detail"].lower()

    # Test empty batch
    response = client.post("/api/v1/embed/batch", json={"documents": []})
    assert response.status_code == 422

    # Test invalid document in batch
    invalid_batch = {
        "documents": [
            {"id": 1, "text": ""},  # Empty text
            {"id": 2, "text": "Valid document"}
        ]
    }
    response = client.post("/api/v1/embed/batch", json=invalid_batch)
    assert response.status_code == 422
    assert "document 1" in response.json()["detail"].lower()

@pytest.fixture
def mock_gpu_cleanup(monkeypatch):
    cleanup_called = False
    def mock_cleanup():
        nonlocal cleanup_called
        cleanup_called = True
    
    monkeypatch.setattr("embed_endpoint.EmbeddingService.cleanup_gpu_memory", mock_cleanup)
    return lambda: cleanup_called

def test_gpu_cleanup(client, mock_gpu_cleanup, sample_text):
    # Test cleanup after large text
    long_text = sample_text * 100  # Make text long enough to trigger cleanup
    response = client.post(
        "/api/v1/embed/text",
        data=long_text,
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 200
    assert mock_gpu_cleanup()  # Verify cleanup was called
