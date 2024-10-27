import pytest
from fastapi.testclient import TestClient
from embed_endpoint import app, Settings, EmbeddingService
from sentence_transformers import SentenceTransformer
import os
import shutil

@pytest.fixture
def test_settings():
    return Settings(
        transformer_model_name="sentence-transformers/all-mpnet-base-v2",
        max_words=350
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
    yield
    # Clean up any test artifacts
    if os.path.exists("tests/test_data/temp"):
        shutil.rmtree("tests/test_data/temp")
