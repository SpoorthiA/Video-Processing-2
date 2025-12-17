import os
import sys
import logging
import ffmpeg
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import shutil
import json

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

# Add project root to path
sys.path.append(os.getcwd())

# Setup FFmpeg Path
possible_ffmpeg_paths = [
    Path("tools/ffmpeg/ffmpeg-7.0-full_build/bin"),      # If running from root
    Path("../tools/ffmpeg/ffmpeg-7.0-full_build/bin"),   # If running from video-retrieval-poc
]

tools_ffmpeg_bin = None
for p in possible_ffmpeg_paths:
    if p.exists():
        tools_ffmpeg_bin = p.resolve()
        break

if tools_ffmpeg_bin:
    os.environ["PATH"] = str(tools_ffmpeg_bin) + os.pathsep + os.environ["PATH"]
    print(f"FFmpeg found at: {tools_ffmpeg_bin}")
else:
    print("Warning: FFmpeg not found in tools directory. Ensure it is installed or in PATH.")

from agents.common.config import settings
from agents.common.database import get_db_session
from agents.common.models import Video, Chunk
from agents.embedding_agent.models import EmbeddingFactory

# Logging
logging.basicConfig(level=settings.LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api_agent")

# Global State for Models
class GlobalState:
    model: Optional[object] = None # Using object as it's an adapter
    model_name: str = settings.EMBEDDING_MODEL

state = GlobalState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Loading Default Embedding Model: {state.model_name}...")
    
    # Use Factory to load default model
    state.model = EmbeddingFactory.get_model({"embedding_model": state.model_name})
    state.model.load()
    
    yield
    # Shutdown
    logger.info("Shutting down API...")
    if state.model:
        state.model.unload()
        state.model = None

app = FastAPI(title="Video Retrieval API", lifespan=lifespan)

# --- Schemas ---
class SearchResult(BaseModel):
    video_id: str
    filename: str
    text: str
    start_time: float
    end_time: float
    score: float
    clip_url: str

# --- Endpoints ---

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), config: str = Form(default="{}")):
    try:
        # Ensure inbox exists
        settings.VIDEO_INBOX_FOLDER.mkdir(parents=True, exist_ok=True)
        
        file_path = settings.VIDEO_INBOX_FOLDER / file.filename
        
        # Save config as sidecar json FIRST to avoid race condition with Ingestion Agent
        try:
            config_dict = json.loads(config)
            if config_dict:
                config_path = file_path.with_suffix(file_path.suffix + ".json")
                with open(config_path, "w") as f:
                    json.dump(config_dict, f)
        except Exception as e:
            logger.warning(f"Failed to save config sidecar: {e}")

        # Then save video file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": file.filename, "status": "uploaded"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Video Retrieval API"}


@app.get("/videos")
def list_videos(status: str = Query(None, description="Filter by status"), model: str = Query(None, description="Filter by embedding model")):
    with get_db_session() as db:
        query = db.query(Video)
        if status:
            query = query.filter(Video.status == status)
        
        videos = query.order_by(Video.created_at.desc()).all()
        
        # Filter by model in Python to avoid SQLite JSON complexity
        if model:
            filtered_videos = []
            for v in videos:
                config = v.processing_config or {}
                # Fallback to default settings if not specified in video config
                video_model = config.get("embedding_model", settings.EMBEDDING_MODEL)
                
                if video_model == model:
                    filtered_videos.append(v)
            videos = filtered_videos
        
        return [
            {
                "id": v.id,
                "filename": v.filename,
                "status": v.status,
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "duration": v.duration_seconds,
                "resolution": v.resolution,
                "config": v.processing_config
            }
            for v in videos
        ]

@app.get("/search", response_model=List[SearchResult])
def search(q: str = Query(..., min_length=2), video_id: Optional[str] = Query(None), model: Optional[str] = Query(None)):
    
    # Determine target model
    target_model_name = model if model else settings.EMBEDDING_MODEL
    
    # Dynamic Model Switching
    if state.model is None or state.model_name != target_model_name:
        logger.info(f"Switching model from '{state.model_name}' to '{target_model_name}'")
        
        if state.model:
            state.model.unload()
            
        try:
            state.model = EmbeddingFactory.get_model({"embedding_model": target_model_name})
            state.model.load()
            state.model_name = target_model_name
        except Exception as e:
            logger.error(f"Failed to load model {target_model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model {target_model_name}")

    if not state.model:
        raise HTTPException(status_code=503, detail="Search services not initialized")
        
    # 1. Embed Query
    # Use .embed() from adapter which returns List[float]
    query_vector = state.model.embed(q)
    
    # 2. Search Qdrant
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        if settings.QDRANT_URL:
            client = QdrantClient(url=settings.QDRANT_URL)
        else:
            client = QdrantClient(path=settings.QDRANT_PATH)
            
        # Construct Filter if video_id is provided
        search_filter = None
        if video_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )

        # Helper to get collection name (matching EmbeddingProcessor logic)
        def get_collection_name(base_name, model_name):
            safe_model_name = model_name.replace("/", "-").replace(".", "-").lower()
            return f"{base_name}_{safe_model_name}"

        # Determine collection names based on the loaded model
        # Note: This assumes the search model matches the ingestion model.
        # For visual search with CLIP/SigLIP, we would need a different search strategy.
        current_model_name = state.model_name
        dialogue_collection = get_collection_name(settings.QDRANT_DIALOGUE_COLLECTION, current_model_name)
        
        # For visual collection, we assume text-based captions (using the same embedding model)
        # If using CLIP/SigLIP, this logic would need to be expanded to load the correct model for query embedding.
        visual_collection = get_collection_name(settings.QDRANT_VISUAL_COLLECTION, current_model_name)

        logger.info(f"Searching in collections: {dialogue_collection}, {visual_collection}")

        # Helper to safely search a collection
        def safe_search(collection_name, boost=1.0, type_label="dialogue"):
            try:
                hits = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=5
                )
                return [{"hit": hit, "score": hit.score * boost, "type": type_label} for hit in hits]
            except Exception as e:
                # Log the error to help debugging (e.g. collection not found)
                logger.warning(f"Search failed for {collection_name}: {e}")
                return []

        # Search both streams
        results_dialogue = safe_search(dialogue_collection, boost=1.0, type_label="dialogue")
        results_visual = safe_search(visual_collection, boost=1.2, type_label="visual")
        
        # Combine and Sort
        all_results = results_dialogue + results_visual
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Deduplicate
        seen = set()
        unique_results = []
        for item in all_results:
            payload = item["hit"].payload
            key = f"{payload.get('video_id')}_{int(payload.get('start_time'))}"
            if key not in seen:
                seen.add(key)
                unique_results.append(item)
                if len(unique_results) >= 5:
                    break
                    
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    response = []
    for item in unique_results:
        hit = item["hit"]
        payload = hit.payload
        video_id = payload.get("video_id")
        start = payload.get("start_time")
        end = payload.get("end_time")
        
        clip_url = f"/clip/{video_id}?start={start}&end={end}"
        
        origin_text = payload.get("text", "")
        if item["type"] == "visual" and not origin_text.startswith("[Visual]"):
             display_text = f"👁️ {origin_text}"
        elif item["type"] == "dialogue":
             display_text = f"💬 {origin_text}"
        else:
             display_text = origin_text

        response.append(SearchResult(
            video_id=video_id,
            filename=payload.get("filename"),
            text=display_text,
            start_time=start,
            end_time=end,
            score=item["score"],
            clip_url=clip_url
        ))
        
    return response

@app.get("/clip/{video_id}")
def get_clip(video_id: str, start: float, end: float):
    """
    Dynamically cuts a clip from the source video and returns it.
    """
    with get_db_session() as db:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        source_path = Path(video.file_path)
        if not source_path.exists():
             raise HTTPException(status_code=404, detail="Source video file missing")

        # Ensure clip output dir exists
        settings.CLIP_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        
        # Unique clip name
        clip_filename = f"{video_id}_{start:.2f}_{end:.2f}.mp4"
        clip_path = settings.CLIP_OUTPUT_FOLDER / clip_filename
        
        # If cached, return it
        if clip_path.exists():
            return FileResponse(clip_path, media_type="video/mp4", filename=clip_filename)
        
        # Generate Clip using FFmpeg
        try:
            logger.info(f"Generating clip for {video.filename} ({start}-{end})")
            (
                ffmpeg
                .input(str(source_path), ss=start, t=end-start)
                .output(str(clip_path), c="copy") # Stream copy is fast/lossless but requires keyframes.
                # If 'c="copy"' is inaccurate, remove it to re-encode (slower but precise).
                # For POC, we'll try re-encoding for precision if copy fails or is inaccurate?
                # Let's switch to re-encoding (default) for better precision on cuts, 
                # but use 'fast' preset.
                .overwrite_output()
                .run(quiet=True)
            )
            return FileResponse(clip_path, media_type="video/mp4", filename=clip_filename)
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise HTTPException(status_code=500, detail="Could not generate clip")

# Mount frontend
# Resolve absolute path to frontend relative to this file
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"

if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    logger.error(f"Frontend directory not found: {frontend_dir}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
