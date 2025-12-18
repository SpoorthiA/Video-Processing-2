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
from agents.common.database import get_db_session, get_transcript_for_video, get_captions_for_video
from agents.common.models import Video, Transcript, Chunk, VideoCaptions
from agents.common.enums import TextEmbeddingModel

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
            
            # Handle specific model requirements if needed
            trust_remote_code = False
            if self.model_name in [TextEmbeddingModel.NOMIC_EMBED_V1_5, TextEmbeddingModel.BGE_M3]:
                trust_remote_code = True
                
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=trust_remote_code)
            logger.info("Model loaded successfully.")
            
            # Ensure Collection Exists
            self._ensure_collection()

    def _get_collection_name(self, base_name):
        safe_model_name = self.model_name.replace("/", "-").replace(".", "-").lower()
        return f"{base_name}_{safe_model_name}"

    def _ensure_collection(self):
        try:
            client = self.get_qdrant_client()
            collections = client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            dimension = self.model.get_sentence_embedding_dimension()
            
            # Use model-specific collection names
            dialogue_col = self._get_collection_name(self.dialogue_collection)
            visual_col = self._get_collection_name(self.visual_collection)
            
            for name in [dialogue_col, visual_col]:
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

    def _process_stream(self, segments, collection_name, video, source_obj, db, client):
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
            
            transcript_id = source_obj.id if isinstance(source_obj, Transcript) else None
            captions_id = source_obj.id if isinstance(source_obj, VideoCaptions) else None
            
            # DB Record
            new_chunk = Chunk(
                id=chunk_id,
                video_id=video.id,
                transcript_id=transcript_id,
                captions_id=captions_id,
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
                    "type": "visual" if collection_name == self.visual_collection else "dialogue",
                    "model_name": source_obj.model_name if source_obj else "unknown"
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
            
        # Determine active models
        speech_model_name = settings.WHISPER_MODEL
        vision_model_name = settings.VISION_MODEL
        
        transcript = get_transcript_for_video(video_id, speech_model_name)
        captions = get_captions_for_video(video_id, vision_model_name)
        
        if not transcript and not captions:
            logger.error(f"No artifacts found for {video.filename} with models {speech_model_name}, {vision_model_name}.")
            return
            
        # Processing Block
        try:
            self.load_model()
            client = self.get_qdrant_client()
            
            with get_db_session() as db:
                # Re-fetch video for session attachment
                video = db.query(Video).filter(Video.id == video_id).first()
                
                # Determine collection names
                dialogue_col = self._get_collection_name(self.dialogue_collection)
                visual_col = self._get_collection_name(self.visual_collection)

                # Process Dialogue Stream
                if transcript:
                    json_path = Path(transcript.json_file_path)
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Handle new format
                            audio_segments = data.get("segments", [])
                            
                        # Re-fetch transcript for session attachment
                        transcript = db.query(Transcript).filter(Transcript.id == transcript.id).first()
                        self._process_stream(audio_segments, dialogue_col, video, transcript, db, client)
                
                # Process Visual Stream
                if captions:
                    json_path = Path(captions.json_file_path)
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            visual_segments = data.get("segments", [])
                            
                        # Re-fetch captions for session attachment
                        captions = db.query(VideoCaptions).filter(VideoCaptions.id == captions.id).first()
                        self._process_stream(visual_segments, visual_col, video, captions, db, client)

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