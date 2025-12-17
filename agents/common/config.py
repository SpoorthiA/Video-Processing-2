from typing import Optional
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Loads and manages configuration from .env file or environment variables."""
    
    # --- Core Infrastructure ---
    DATABASE_URL: str
    QDRANT_URL: Optional[str] = None
    QDRANT_PATH: str = "qdrant_storage" # Local storage path
    QDRANT_COLLECTION_NAME: str = "video_chunks" # Legacy/Fallback
    QDRANT_DIALOGUE_COLLECTION: str = "dialogue_chunks"
    QDRANT_VISUAL_COLLECTION: str = "visual_chunks"
    
    # --- File Paths ---
    VIDEO_INBOX_FOLDER: Path = Path("./data/inbox")
    VIDEO_PROCESSED_FOLDER: Path = Path("./data/videos")
    CLIP_OUTPUT_FOLDER: Path = Path("./data/clips")
    TRANSCRIPT_FOLDER: Path = Path("./data/transcripts")
    THUMBNAIL_FOLDER: Path = Path("./data/thumbnails")
    
    # --- AI Models ---
    WHISPER_MODEL: str = "base"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE_SECONDS: int = 10
    CHUNK_OVERLAP_SECONDS: int = 2
    
    # --- Agent Settings ---
    INGESTION_AGENT_INTERVAL: int = 10 # seconds between scans
    TRANSCRIPTION_AGENT_INTERVAL: int = 10 # seconds
    
    # --- API Settings ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    
    class Config:
        # Resolve .env path relative to this file (agents/common/config.py -> .../.env)
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        env_file_encoding = 'utf-8'
        extra = "ignore"  # Allow extra fields in .env without crashing

# Create a single, globally accessible settings instance
settings = Settings()