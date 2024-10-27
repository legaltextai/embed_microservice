import os
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException, Request, Body, Response
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import nltk
from nltk.tokenize import sent_tokenize
from fastapi.openapi.utils import get_openapi
import re
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
REQUEST_COUNT = Counter(
    'inception_requests_total',
    'Total number of embedding requests',
    ['endpoint']
)

PROCESSING_TIME = Histogram(
    'inception_processing_seconds',
    'Time spent processing embedding requests',
    ['endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf"))
)

ERROR_COUNT = Counter(
    'inception_errors_total',
    'Total number of errors',
    ['endpoint', 'error_type']
)

CHUNK_COUNT = Counter(
    'inception_chunks_total',
    'Total number of text chunks processed',
    ['endpoint']
)

MODEL_LOAD_TIME = Histogram(
    'inception_model_load_seconds',
    'Time spent loading the model',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, float("inf"))
)

# Download NLTK data at startup
nltk.download('punkt', quiet=True)

class Settings(BaseSettings):
    # Add validation and documentation
    transformer_model_name: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="Name of the transformer model to use"
    )
    max_words: int = Field(
        350,
        ge=1,
        le=1000,
        description="Maximum words per chunk"
    )
    max_text_length: int = 100000  # Maximum text length in characters
    force_cpu: bool = False
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"


class TextRequest(BaseModel):
    id: int
    text: str = Field(
        ..., 
        description="The text content of the opinion",
        example="The Supreme Court's decision in Brown v. Board of Education was a landmark ruling."
    )

class BatchTextRequest(BaseModel):
    documents: List[TextRequest] = Field(
        ..., 
        description="List of documents to process. Each document should have an ID and text content.",
        example=[
            {
                "id": 1,
                "text": """The First Amendment protects freedom of speech and religion.

This fundamental right is crucial to democracy."""
            },
            {
                "id": 2,
                "text": """Marbury v. Madison (1803) established judicial review.

This case expanded judicial power significantly."""
            }
        ]
    )

class ChunkEmbedding(BaseModel):
    chunk_number: int
    chunk: str
    embedding: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_number": 1,
                "chunk": "This is a sample chunk of text from a legal opinion.",
                "embedding": [0.123, 0.456, 0.789]
            }
        }

class TextResponse(BaseModel):
    id: Optional[int] = None
    embeddings: List[ChunkEmbedding]
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "embeddings": [
                    {
                        "chunk_number": 1,
                        "chunk": "First chunk of the legal opinion text.",
                        "embedding": [0.123, 0.456, 0.789]
                    },
                    {
                        "chunk_number": 2,
                        "chunk": "Second chunk of the legal opinion text.",
                        "embedding": [0.321, 0.654, 0.987]
                    }
                ]
            }
        }

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    embedding: List[float]

#although we are doing preprocessing here, we needto decide if we want to do it here or in the client script that wwill be sending opinions for embedding 
def clean_text_for_json(text: str) -> str:
    """
    Clean and prepare text for JSON encoding.
    Handles special characters, line breaks, and other potential JSON issues.
    """
    if not text:
        return ""
    
    try:
        # Remove null bytes and other control characters except newlines and tabs
        text = ''.join(char for char in text if char == '\n' or char == '\t' or (ord(char) >= 32 and ord(char) < 127))
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove spaces at the beginning and end of each line
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        # Remove multiple consecutive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    except Exception as e:
        raise ValueError(f"Error cleaning text: {str(e)}")

def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding generation.
    Includes cleaning and validation steps.
    """
    try:
        cleaned_text = clean_text_for_json(text)
        
        if not cleaned_text:
            raise ValueError("Text is empty after cleaning")
        
        
        return cleaned_text
    except Exception as e:
        raise ValueError(f"Error preprocessing text: {str(e)}")

class EmbeddingService:
    def __init__(self, model: SentenceTransformer, max_words: int):
        start_time = time.time()
        try:
            self.model = model
            device = 'cpu' if Settings().force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
            self.gpu_model = model.to(device)
            self.max_words = max_words
            self.pool = self.gpu_model.start_multi_process_pool()
            MODEL_LOAD_TIME.observe(time.time() - start_time)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise

    def __del__(self):
        try:
            if hasattr(self, 'pool'):
                self.gpu_model.stop_multi_process_pool(self.pool)
                logger.info("Model pool stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping model pool: {str(e)}")

    async def generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single query text"""
        if not text.strip():
            raise ValueError("Empty query text")
        
        try:
            with torch.no_grad():
                embedding = self.model.encode(text, device='cpu')
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks based on sentences, not exceeding max_words"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            if current_word_count + sentence_word_count <= self.max_words:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    async def generate_text_embeddings(self, texts: List[str]) -> List[List[ChunkEmbedding]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            raise ValueError("Empty text list")
            
        try:
            all_embeddings = []
            all_chunks = []
            chunk_counts = []

            for text in texts:
                chunks = self.split_text_into_chunks(text)
                CHUNK_COUNT.labels(endpoint='text').inc(len(chunks))
                all_chunks.extend(chunks)
                chunk_counts.append(len(chunks))
            
            embeddings = self.gpu_model.encode_multi_process(
                sentences=all_chunks,
                pool=self.pool,
                batch_size=8,
                show_progress_bar=False
            )

            start_index = 0
            for count in chunk_counts:
                text_embeddings = []
                for j in range(count):
                    chunk = all_chunks[start_index + j]
                    embedding = embeddings[start_index + j]
                    text_embeddings.append(ChunkEmbedding(
                        chunk_number=j + 1,
                        chunk=chunk,
                        embedding=embedding.tolist()
                    ))
                all_embeddings.append(text_embeddings)
                start_index += count

            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            raise

    def cleanup_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Inception v2",
    description="Service for generating embeddings from queries and opinions",
    version="2.0.0"
)

embedding_service: Optional[EmbeddingService] = None
settings: Settings = Settings()

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=os.getenv("ALLOWED_METHODS", "*").split(","),
    allow_headers=os.getenv("ALLOWED_HEADERS", "*").split(","),
)

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model and service on startup"""
    global embedding_service
    try:
        settings = Settings()
        model = SentenceTransformer(settings.transformer_model_name)
        embedding_service = EmbeddingService(model=model, max_words=settings.max_words)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise RuntimeError(f"Failed to initialize embedding service: {str(e)}")

# Add new heartbeat endpoint
@app.get("/")
async def heartbeat():
    """Simple heartbeat endpoint"""
    return "Heartbeat detected."

# Update health check endpoint
@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy" if embedding_service else "service_unavailable",
        "model_loaded": embedding_service is not None,
        "gpu_available": gpu_available and not Settings().force_cpu
    }

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not Settings().enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Update existing endpoints to include metrics and error tracking
@app.post("/api/v1/embed/query", response_model=QueryResponse)
async def create_query_embedding(request: QueryRequest):
    """Generate embedding for a query"""
    REQUEST_COUNT.labels(endpoint='query').inc()
    start_time = time.time()
    
    if not embedding_service:
        ERROR_COUNT.labels(endpoint='query', error_type='service_unavailable').inc()
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        embedding = await embedding_service.generate_query_embedding(request.text)
        PROCESSING_TIME.labels(endpoint='query').observe(time.time() - start_time)
        return QueryResponse(embedding=embedding)
    except ValueError as e:
        ERROR_COUNT.labels(endpoint='query', error_type='validation_error').inc()
        raise HTTPException(status_code=422, detail=str(e))
    except torch.cuda.OutOfMemoryError as e:
        ERROR_COUNT.labels(endpoint='query', error_type='gpu_error').inc()
        raise HTTPException(status_code=503, detail="GPU memory exhausted")
    except Exception as e:
        ERROR_COUNT.labels(endpoint='query', error_type='processing_error').inc()
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/v1/embed/text", response_model=TextResponse)
async def create_text_embedding(request: Request):
    """Generate embeddings for opinion text"""
    REQUEST_COUNT.labels(endpoint='text').inc()
    start_time = time.time()

    if not embedding_service:
        ERROR_COUNT.labels(endpoint='text', error_type='service_unavailable').inc()
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        raw_text = await request.body()
        text = raw_text.decode("utf-8")
        
        if not text.strip():
            ERROR_COUNT.labels(endpoint='text', error_type='empty_input').inc()
            raise ValueError("Empty text input")

        if len(text) > Settings().max_text_length:
            ERROR_COUNT.labels(endpoint='text', error_type='text_too_long').inc()
            raise ValueError(f"Text exceeds maximum length of {Settings().max_text_length} characters")

        result = await embedding_service.generate_text_embeddings([text])
        PROCESSING_TIME.labels(endpoint='text').observe(time.time() - start_time)
        return TextResponse(embeddings=result[0])
    except Exception as e:
        ERROR_COUNT.labels(endpoint='text', error_type='processing_error').inc()
        sentry_sdk.capture_exception(e)
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.post("/api/v1/embed/batch", response_model=List[TextResponse])
async def create_batch_text_embeddings(request: BatchTextRequest):
    """Generate embeddings for multiple documents"""
    REQUEST_COUNT.labels(endpoint='batch').inc()
    start_time = time.time()
    
    if not embedding_service:
        ERROR_COUNT.labels(endpoint='batch', error_type='service_unavailable').inc()
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        texts = [doc.text for doc in request.documents]
        embeddings_list = await embedding_service.generate_text_embeddings(texts)
        
        results = [
            TextResponse(id=doc.id, embeddings=embeddings)
            for doc, embeddings in zip(request.documents, embeddings_list)
        ]
        
        PROCESSING_TIME.labels(endpoint='batch').observe(time.time() - start_time)
        return results
    except Exception as e:
        ERROR_COUNT.labels(endpoint='batch', error_type='processing_error').inc()
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

# this is a temporary validation endpoint to test text preprocessing
@app.post("/api/v1/validate/text")
async def validate_text(request: TextRequest):
    """
    Validate and clean text without generating embeddings.
    Useful for testing text preprocessing.
    """
    try:
        processed_text = preprocess_text(request.text)
        return {
            "id": request.id,
            "original_text": request.text,
            "processed_text": processed_text,
            "is_valid": True
        }
    except Exception as e:
        return {
            "id": request.id,
            "original_text": request.text,
            "error": str(e),
            "is_valid": False
        }

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Inception v2",
        version="2.0.0",
        description="Service for generating embeddings from queries and opinions",
        routes=app.routes,
    )

    for path in openapi_schema["paths"]:
        if path == "/api/v1/embed/text":
            openapi_schema["paths"][path]["post"]["requestBody"] = {
                "content": {
                    "text/plain": {
                        "example": "A very long opinion goes here.\nIt can span multiple lines.\nEach line will be preserved."
                    }
                },
                "required": True
            }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding generation.
    Includes cleaning and validation steps.
    """
    try:
        cleaned_text = clean_text_for_json(text)
        
        if not cleaned_text:
            raise ValueError("Text is empty after cleaning")
        
        return cleaned_text
    except Exception as e:
        raise ValueError(f"Error preprocessing text: {str(e)}")

app.openapi = custom_openapi

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global embedding_service
    try:
        if embedding_service:
            embedding_service.__del__()
            embedding_service = None
    except Exception as e:
        sentry_sdk.capture_exception(e)
        logger.error(f"Error during shutdown: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 

    
















