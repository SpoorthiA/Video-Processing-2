import json
import logging
import torch
import uuid
from pathlib import Path
from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as RestModels
from qdrant_client.models import PointStruct, VectorParams, Distance

from agents.common.config import settings
from agents.common.database import get_db_session
from agents.common.models import Video, Transcript, Chunk

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.EMBEDDING_MODEL
        self.model = None
        self.qdrant = None
        
        # Load Qdrant Client (Local Mode or Server)
        # We do NOT hold the connection open in __init__ for local mode to avoid locking issues
        # with the API process. We will connect on demand.
        self.qdrant_url = settings.QDRANT_URL
        self.qdrant_path = settings.QDRANT_PATH
            
        self.dialogue_collection = settings.QDRANT_DIALOGUE_COLLECTION
        self.visual_collection = settings.QDRANT_VISUAL_COLLECTION
        self.chunk_size = settings.CHUNK_SIZE_SECONDS
        self.chunk_overlap = settings.CHUNK_OVERLAP_SECONDS
        
        logger.info(f"Embedding Agent initialized. Device: {self.device}")

    def get_qdrant_client(self):
        if self.qdrant_url:
            return QdrantClient(url=self.qdrant_url)
        else:
            return QdrantClient(path=self.qdrant_path)

    def load_model(self):
        if not self.model:
            logger.info(f"Loading Embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully.")
            
            # Ensure Collection Exists
            self._ensure_collection()

    def _ensure_collection(self):
        try:
            client = self.get_qdrant_client()
            collections = client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            dimension = self.model.get_sentence_embedding_dimension()
            
            for name in [self.dialogue_collection, self.visual_collection]:
                if name not in existing_names:
                    logger.info(f"Creating Qdrant collection: {name}")
                    client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                    )
        except Exception as e:
            logger.error(f"Error checking/creating collection: {e}")

    def create_chunks(self, segments):
        """
        Groups Whisper segments into time-based chunks.
        """
        chunks = []
        current_chunk_segments = []
        current_chunk_start = 0.0
        
        # Simple greedy chunking based on time
        # This can be improved with overlaps, but for POC we do sequential blocks
        
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            
            if not current_chunk_segments:
                current_chunk_start = seg_start
            
            current_chunk_segments.append(seg)
            
            # Check if chunk is long enough
            if (seg_end - current_chunk_start) >= self.chunk_size:
                chunks.append(self._finalize_chunk(current_chunk_segments))
                current_chunk_segments = []
                # Overlap logic could go here (retain last segment)

        # Add remaining segments
        if current_chunk_segments:
             chunks.append(self._finalize_chunk(current_chunk_segments))
             
        return chunks

    def _finalize_chunk(self, segments):
        text = " ".join([s["text"].strip() for s in segments])
        start = segments[0]["start"]
        end = segments[-1]["end"]
        return {
            "text": text,
            "start": start,
            "end": end,
            "duration": end - start
        }

    def _process_stream(self, segments, collection_name, video, transcript, db, client):
        if not segments:
            return

        chunks_data = self.create_chunks(segments)
        if not chunks_data:
            return

        texts = [c["text"] for c in chunks_data]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        points = []
        for i, chunk_data in enumerate(chunks_data):
            chunk_id = str(uuid.uuid4())
            
            # DB Record
            new_chunk = Chunk(
                id=chunk_id,
                video_id=video.id,
                transcript_id=transcript.id,
                chunk_index=i,
                text=chunk_data["text"],
                start_time_seconds=chunk_data["start"],
                end_time_seconds=chunk_data["end"],
                duration_seconds=chunk_data["duration"],
                embedding_id=chunk_id
            )
            db.add(new_chunk)
            
            # Qdrant Point
            points.append(PointStruct(
                id=chunk_id,
                vector=embeddings[i].tolist(),
                payload={
                    "video_id": str(video.id),
                    "filename": video.filename,
                    "text": chunk_data["text"],
                    "start_time": chunk_data["start"],
                    "end_time": chunk_data["end"],
                    "type": "visual" if collection_name == self.visual_collection else "dialogue"
                }
            ))

        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Indexed {len(points)} vectors to {collection_name}.")

    def process_video(self, video_id):
        with get_db_session() as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video:
                return

            logger.info(f"Embedding processing for: {video.filename}")
            
            # Get transcript
            transcript = db.query(Transcript).filter(Transcript.video_id == video.id).first()
            if not transcript or not transcript.json_file_path:
                logger.error("No transcript found for video.")
                return
            
            json_path = Path(transcript.json_file_path)
        
        # Processing Block
        try:
            self.load_model()
            
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Fallback to 'segments' if specific keys missing (backward compatibility)
                audio_segments = data.get("audio_segments", [])
                visual_segments = data.get("visual_segments", [])
                
                if not audio_segments and not visual_segments:
                    # Try legacy format
                    all_segments = data.get("segments", [])
                    audio_segments = [s for s in all_segments if s.get("type") != "visual"]
                    visual_segments = [s for s in all_segments if s.get("type") == "visual"]

            client = self.get_qdrant_client()
            
            with get_db_session() as db:
                # Re-fetch for session attachment
                video = db.query(Video).filter(Video.id == video_id).first()
                transcript = db.query(Transcript).filter(Transcript.video_id == video.id).first()
                
                # Process Dialogue Stream
                self._process_stream(audio_segments, self.dialogue_collection, video, transcript, db, client)
                
                # Process Visual Stream
                self._process_stream(visual_segments, self.visual_collection, video, transcript, db, client)

                video.status = "ready"
                db.commit()
                logger.info(f"Processing complete for {video.filename}.")

        except Exception as e:
            logger.error(f"Error embedding {video.filename}: {e}", exc_info=True)

    def run_once(self):
        video_id = None
        with get_db_session() as db:
            video = db.query(Video).filter(Video.status == "pending_embedding").first()
            if video:
                video_id = video.id
        
        if video_id:
            self.process_video(video_id)
            return 1
        return 0