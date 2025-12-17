import os
import json
import logging
import torch
import datetime
import cv2
from PIL import Image
from pathlib import Path

# --- PATH SETUP FOR FFMPEG ---
# Must be done before importing libraries that use ffmpeg
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

from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration

from agents.common.config import settings
from agents.common.database import get_db_session
from agents.common.models import Video, Transcript

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use int8 for CPU to be faster/compatible, float16 for GPU
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        self.whisper_model_name = settings.WHISPER_MODEL
        self.whisper_model = None
        
        self.vision_model_name = "Salesforce/blip-image-captioning-base"
        self.vision_processor = None
        self.vision_model = None
        
        # Ensure transcript folder exists
        settings.TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Transcription Agent initialized. Device: {self.device}, Compute: {self.compute_type}")

    def load_whisper_model(self):
        if not self.whisper_model:
            logger.info(f"Loading Faster-Whisper model: {self.whisper_model_name}...")
            try:
                self.whisper_model = WhisperModel(
                    self.whisper_model_name, 
                    device=self.device, 
                    compute_type=self.compute_type
                )
                logger.info("Whisper Model loaded successfully.")
            except Exception as e:
                logger.critical(f"Failed to load Whisper model: {e}")
                raise

    def load_vision_model(self):
        if not self.vision_model:
            logger.info(f"Loading Vision model: {self.vision_model_name}...")
            try:
                self.vision_processor = BlipProcessor.from_pretrained(self.vision_model_name)
                self.vision_model = BlipForConditionalGeneration.from_pretrained(self.vision_model_name).to(self.device)
                logger.info("Vision Model loaded successfully.")
            except Exception as e:
                logger.critical(f"Failed to load Vision model: {e}")
                raise

    def process_visuals(self, video_path):
        """
        Extracts frames and generates captions for visual events.
        """
        self.load_vision_model()
        
        visual_segments = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 24 # Fallback
        
        interval_seconds = 5 # Analyze every 5 seconds
        frame_interval = int(fps * interval_seconds)
        frame_count = 0
        
        logger.info(f"Starting visual analysis for {video_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Generate Caption
                    inputs = self.vision_processor(pil_image, return_tensors="pt").to(self.device)
                    out = self.vision_model.generate(**inputs, max_new_tokens=50)
                    caption = self.vision_processor.decode(out[0], skip_special_tokens=True)
                    
                    timestamp = frame_count / fps
                    
                    visual_segments.append({
                        "start": timestamp,
                        "end": timestamp + interval_seconds,
                        "text": f"[Visual]: {caption}",
                        "type": "visual"
                    })
                except Exception as e:
                    logger.warning(f"Failed to process frame at {frame_count}: {e}")
                
            frame_count += 1
            
        cap.release()
        logger.info(f"Visual analysis complete. Found {len(visual_segments)} visual events.")
        return visual_segments

    def process_video(self, video_id):
        # We fetch the video inside the method to ensure fresh state
        with get_db_session() as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video:
                logger.error(f"Video ID {video_id} not found in database.")
                return
            
            file_path = video.file_path
            filename = video.filename
            
        # Processing Block (No DB session active)
        try:
            self.load_whisper_model()
            
            logger.info(f"Transcribing {filename}...")
            
            # 1. Transcribe
            # segments is a generator
            segments_generator, info = self.whisper_model.transcribe(file_path, beam_size=5)
            
            # Convert generator to list to iterate multiple times if needed and for saving
            segments = []
            full_text_parts = []
            
            for segment in segments_generator:
                seg_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "type": "audio"
                }
                segments.append(seg_data)
                full_text_parts.append(segment.text.strip())

            # 2. Visual Analysis
            visual_segments = self.process_visuals(file_path)
            
            # 3. Merge and Sort
            all_segments = segments + visual_segments
            all_segments.sort(key=lambda x: x["start"])

            # Prepare Paths
            base_name = Path(filename).stem
            json_path = settings.TRANSCRIPT_FOLDER / f"{base_name}.json"
            vtt_path = settings.TRANSCRIPT_FOLDER / f"{base_name}.vtt"
            
            # Save JSON
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": all_segments, # Keep for backward compatibility
                "audio_segments": segments,
                "visual_segments": visual_segments
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            
            # Save VTT (Only audio segments for standard players)
            self.save_vtt(segments, vtt_path)
            
            # Calculate full text (Include visuals for searchability)
            # We construct a rich text representation
            rich_text_parts = [s["text"].strip() for s in all_segments]
            full_text = " ".join(rich_text_parts)
            word_count = len(full_text.split())

            # Update Database (New Session)
            with get_db_session() as db:
                video = db.query(Video).filter(Video.id == video_id).first()
                if video:
                    new_transcript = Transcript(
                        video_id=video.id,
                        full_text=full_text,
                        vtt_file_path=str(vtt_path),
                        json_file_path=str(json_path),
                        word_count=word_count
                    )
                    db.add(new_transcript)
                    video.status = "pending_embedding"
                    db.commit()
                    logger.info(f"Completed transcription for {filename}. Status updated.")

        except Exception as e:
            logger.error(f"Error transcribing {filename}: {e}", exc_info=True)
            # Optional: Mark as error in DB to avoid infinite retries
            # with get_db_session() as db:
            #     video = db.query(Video).filter(Video.id == video_id).first()
            #     video.status = "error_transcription"
            #     db.commit()

    def save_vtt(self, segments, path):
        """Simple VTT writer"""
        def format_time(seconds):
            # VTT format: HH:MM:SS.mmm
            total_seconds = int(seconds)
            milliseconds = int((seconds - total_seconds) * 1000)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"

        with open(path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = format_time(seg["start"])
                end = format_time(seg["end"])
                text = seg["text"].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    def run_once(self):
        """Polls for one pending video and processes it."""
        video_id = None
        with get_db_session() as db:
            video = db.query(Video).filter(Video.status == "pending_transcription").first()
            if video:
                video_id = video.id
        
        if video_id:
            self.process_video(video_id)
            return 1
        
        return 0