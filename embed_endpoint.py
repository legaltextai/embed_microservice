import os
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException, Request, Body
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import nltk
from nltk.tokenize import sent_tokenize
from fastapi.openapi.utils import get_openapi
import re
import numpy as np
import json

nltk.download('punkt')

class Settings(BaseSettings):
    transformer_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    max_words: int = 350
    
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
        self.model = model
        self.gpu_model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_words = max_words
        self.pool = self.gpu_model.start_multi_process_pool()

    def __del__(self):
        if hasattr(self, 'pool'):
            self.gpu_model.stop_multi_process_pool(self.pool)

    def generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single query text using CPU"""
        with torch.no_grad():
            embedding = self.model.encode(text, device='cpu')  #explicitly use CPU only for queries 
        return embedding.tolist()

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

    def generate_text_embeddings(self, texts: List[str]) -> List[List[ChunkEmbedding]]:
        all_embeddings = []
        all_chunks = []
        chunk_counts = []

        for text in texts:
            chunks = self.split_text_into_chunks(text)
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))
        
        try:
            # Use encode_multi_process for batch processing
            embeddings = self.gpu_model.encode_multi_process(
                sentences=all_chunks,
                pool=self.pool,
                batch_size=8,
                show_progress_bar=False
            )
        except Exception as e:
            raise Exception(f"Error during encoding: {str(e)}")

        try:
            start_index = 0
            for i, count in enumerate(chunk_counts):
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
        except Exception as e:
            raise Exception(f"Error processing embeddings: {str(e)}")

        return all_embeddings


app = FastAPI(
    title="Inception v2",
    description="Service for generating embeddings from queries and opinions",
    version="2.0.0"
)

embedding_service: Optional[EmbeddingService] = None
settings: Settings = Settings()

@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model and service on startup"""
    global embedding_service
    model = SentenceTransformer(settings.transformer_model_name)
    embedding_service = EmbeddingService(model=model, max_words=settings.max_words)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": embedding_service is not None
    }

@app.post("/api/v1/embed/query", response_model=QueryResponse)
async def create_query_embedding(request: QueryRequest):
    """Generate embedding for a single query text using CPU"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    
    try:
        embedding = embedding_service.generate_query_embedding(request.text)
        return QueryResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/v1/embed/text", response_model=TextResponse)
async def create_text_embedding(request: Request):
    """
    Generate embeddings for opinion text input.
    """
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        raw_text = await request.body()
        text = raw_text.decode("utf-8")

        result = embedding_service.generate_text_embeddings([text])
        return TextResponse(embeddings=result[0])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.post("/api/v1/embed/batch", response_model=List[TextResponse])
async def create_batch_text_embeddings(request: BatchTextRequest):
    """
    Generate embeddings for a batch of documents.
    """
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        results = []
        for doc in request.documents:
            processed_text = preprocess_text(doc.text)
            embeddings = embedding_service.generate_text_embeddings([processed_text])[0]
            results.append(TextResponse(
                id=doc.id,
                embeddings=embeddings
            ))
        return results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005) 

    
