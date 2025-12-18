import uuid
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    BigInteger,
    JSON,
    UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base  # Import Base from our new database.py

def generate_uuid():
    return str(uuid.uuid4())

class Video(Base):
    __tablename__ = "videos"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), unique=True, nullable=False)
    file_path = Column(Text, nullable=False)
    duration_seconds = Column(Float, nullable=False)
    file_size_bytes = Column(BigInteger)
    fps = Column(Float)
    resolution = Column(String(50))
    language = Column(String(10), default="en")
    status = Column(String(50), default="new", nullable=False)
    processing_config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    transcripts = relationship("Transcript", back_populates="video", cascade="all, delete-orphan")
    captions = relationship("VideoCaptions", back_populates="video", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="video", cascade="all, delete-orphan")

class Transcript(Base):
    __tablename__ = "transcripts"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    full_text = Column(Text)
    vtt_file_path = Column(Text)
    json_file_path = Column(Text)
    word_count = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (UniqueConstraint('video_id', 'model_name', name='uq_video_transcript_model'),)
    
    video = relationship("Video", back_populates="transcripts")
    chunks = relationship("Chunk", back_populates="transcript", cascade="all, delete-orphan")

class VideoCaptions(Base):
    __tablename__ = "video_captions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    full_text = Column(Text)
    json_file_path = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (UniqueConstraint('video_id', 'model_name', name='uq_video_captions_model'),)
    
    video = relationship("Video", back_populates="captions")
    chunks = relationship("Chunk", back_populates="captions", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    transcript_id = Column(String(36), ForeignKey("transcripts.id"), nullable=True)
    captions_id = Column(String(36), ForeignKey("video_captions.id"), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    start_time_seconds = Column(Float, nullable=False)
    end_time_seconds = Column(Float, nullable=False)
    duration_seconds = Column(Float)
    embedding_id = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    video = relationship("Video", back_populates="chunks")
    transcript = relationship("Transcript", back_populates="chunks")
    captions = relationship("VideoCaptions", back_populates="chunks")

class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    query = Column(Text, nullable=False)
    query_embedding_id = Column(String(255))
    results_count = Column(Integer)
    top_result_chunk_id = Column(String(36), ForeignKey("chunks.id"))
    execution_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ClipExport(Base):
    __tablename__ = "clip_exports"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    chunk_id = Column(String(36), ForeignKey("chunks.id"), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())