from pydantic import BaseModel
from typing import Optional
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application settings."""
    
    # Anthropic (Claude) API Configuration
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "claude-sonnet-4-20250514")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    # Retriever Configuration
    index_path: Optional[str] = os.getenv("INDEX_PATH", "./data/index.faiss")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "dmis-lab/biobert-base-cased-v1.2")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Data Sources
    pubmed_api_key: Optional[str] = os.getenv("PUBMED_API_KEY")
    orphanet_api_key: Optional[str] = os.getenv("ORPHANET_API_KEY")
    pubmed_email: str = os.getenv("PUBMED_EMAIL", "nvpatil@usc.edu")
    pubmed_tool_name: str = "RAG_Medical_Assistant"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings() 