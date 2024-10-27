import pytest
from fastapi.testclient import TestClient
from embed_endpoint import app, Settings, EmbeddingService
from sentence_transformers import SentenceTransformer
import torch

# Basic fixtures and setup
@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_text():
    with open("tests/test_data/sample_opinion.txt") as f:
        return f.read()

# Health and basic endpoint tests
def test_health_check(client):
    """Test health check endpoint returns correct status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "gpu_available" in data

def test_heartbeat(client):
    """Test heartbeat endpoint is responding"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == '"Heartbeat detected."'

# Query embedding tests
def test_query_embedding(client):
    """Test basic query embedding functionality"""
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

# Input validation tests
def test_query_embedding_validation(client):
    """Test query endpoint input validation"""
    settings = Settings()
    test_cases = [
        {
            "name": "empty text",
            "input": {"text": ""},
            "expected_status": 422,
            "expected_error": "text length (0) below minimum"
        },
        {
            "name": "text too long",
            "input": {"text": "a" * (settings.max_text_length + 1)},
            "expected_status": 422,
            "expected_error": "text length"
        }
    ]
    
    for case in test_cases:
        response = client.post("/api/v1/embed/query", json=case["input"])
        assert response.status_code == case["expected_status"], f"Failed on: {case['name']}"
        assert case["expected_error"] in response.json()["detail"].lower()

def test_text_embedding_validation(client):
    """Test text endpoint input validation"""
    settings = Settings()
    test_cases = [
        {
            "name": "empty text",
            "input": "",
            "expected_status": 422,
            "expected_error": "text length (0) below minimum"
        },
        {
            "name": "text too long",
            "input": "a" * (settings.max_text_length + 1),
            "expected_status": 422,
            "expected_error": "text length"
        },
        {
            "name": "invalid UTF-8",
            "input": bytes([0xFF, 0xFE, 0xFD]),
            "expected_status": 422,
            "expected_error": "invalid utf-8"
        }
    ]
    
    for case in test_cases:
        response = client.post(
            "/api/v1/embed/text",
            data=case["input"],
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == case["expected_status"], f"Failed on: {case['name']}"
        assert case["expected_error"] in response.json()["detail"].lower()

def test_service_unavailable(client, monkeypatch):
    monkeypatch.setattr("embed_endpoint.embedding_service", None)
    response = client.post(
        "/api/v1/embed/query",
        json={"text": "test"}
    )
    assert response.status_code == 503
    assert "service not initialized" in response.json()["detail"].lower()

# Batch processing tests
def test_batch_processing(client):
    """Test successful batch processing"""
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
    """Test batch processing validation"""
    settings = Settings()
    test_cases = [
        {
            "name": "batch size limit",
            "input": {
                "documents": [
                    {"id": i, "text": f"Document {i}"}
                    for i in range(settings.max_batch_size + 1)
                ]
            },
            "expected_status": 422,
            "expected_error": "batch size exceeds maximum"
        },
        {
            "name": "empty batch",
            "input": {"documents": []},
            "expected_status": 422,
            "expected_error": "empty text list"
        },
        {
            "name": "invalid document",
            "input": {
                "documents": [
                    {"id": 1, "text": ""},  # Empty text
                    {"id": 2, "text": "Valid document"}
                ]
            },
            "expected_status": 422,
            "expected_error": "document 1"
        }
    ]
    
    for case in test_cases:
        response = client.post("/api/v1/embed/batch", json=case["input"])
        assert response.status_code == case["expected_status"], f"Failed on: {case['name']}"
        assert case["expected_error"] in response.json()["detail"].lower()

# GPU memory management tests
@pytest.fixture
def mock_gpu_cleanup(monkeypatch):
    """Fixture to track GPU memory cleanup calls"""
    cleanup_called = False
    def mock_cleanup():
        nonlocal cleanup_called
        cleanup_called = True
    
    monkeypatch.setattr("embed_endpoint.EmbeddingService.cleanup_gpu_memory", mock_cleanup)
    return lambda: cleanup_called

@pytest.mark.gpu
def test_gpu_cleanup(client, mock_gpu_cleanup, sample_text):
    """Test GPU memory cleanup after processing large texts"""
    # Test cleanup after large text
    long_text = sample_text * 100  # Make text long enough to trigger cleanup
    response = client.post(
        "/api/v1/embed/text",
        data=long_text,
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 200
    assert mock_gpu_cleanup(), "GPU memory cleanup was not called"

# Text processing tests
@pytest.mark.text_processing
def test_text_chunking(test_service, sample_text):
    """Test that text chunking works correctly with real-world text"""
    # Get chunks
    chunks = test_service.split_text_into_chunks(sample_text)
    
    # Verify basic properties
    assert len(chunks) > 0, "No chunks were generated"
    assert all(isinstance(chunk, str) for chunk in chunks), "Non-string chunk found"
    assert all(len(chunk.split()) <= test_service.max_words for chunk in chunks), \
        "Chunk exceeds maximum word limit"
    
    # Verify that chunks maintain sentence boundaries
    for i, chunk in enumerate(chunks):
        # Each chunk should end with proper punctuation
        assert chunk.strip()[-1] in {'.', '?', '!', '"'}, \
            f"Chunk {i} does not end with proper punctuation: {chunk[-10:]}"
        # Each chunk should be a complete sentence or sentences
        assert all(sent.strip() for sent in chunk.split('.') if sent.strip()), \
            f"Chunk {i} contains incomplete sentences"
    
    # Verify that all content is preserved (ignoring whitespace)
    original_content = ''.join(sample_text.split())
    chunked_content = ''.join(''.join(chunks).split())
    assert original_content == chunked_content, "Content was lost or altered during chunking"

    # Verify chunk transitions
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i].strip()
        next_chunk = chunks[i + 1].strip()
        assert current_chunk[-1] in {'.', '?', '!', '"'}, \
            f"Chunk {i} does not end with proper punctuation"
        assert next_chunk[0].isupper(), \
            f"Chunk {i + 1} does not start with uppercase letter"
