import pytest
from fastapi.testclient import TestClient
from embed_endpoint import app, Settings, EmbeddingService
from sentence_transformers import SentenceTransformer
import torch
import os
import shutil

@pytest.fixture
def test_settings():
    return Settings(
        transformer_model_name="sentence-transformers/all-mpnet-base-v2",
        max_words=350,
        max_text_length=100000,
        min_text_length=1,
        max_batch_size=100,
        pool_timeout=3600,
        force_cpu=True,  # Force CPU for tests
        enable_metrics=True
    )

@pytest.fixture
def test_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@pytest.fixture
def test_service(test_model, test_settings):
    return EmbeddingService(model=test_model, max_words=test_settings.max_words)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    # Setup: ensure test directories exist
    os.makedirs("tests/test_data/temp", exist_ok=True)
    
    yield
    
    # Cleanup: remove test artifacts and clear GPU memory
    if os.path.exists("tests/test_data/temp"):
        shutil.rmtree("tests/test_data/temp")
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

