"""
Configuration settings for RAGOps Platform.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_title: str = "RAGOps Platform"
    api_version: str = "1.0.0"
    api_description: str = "A platform for RAG operations and insights"
    
    # Database Configuration
    database_url: str = "sqlite:///rag_store.sqlite"
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"


# Create global settings instance
settings = Settings()
