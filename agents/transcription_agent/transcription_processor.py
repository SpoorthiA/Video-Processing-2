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
]

tools_ffmpeg_bin = None
for p in possible_ffmpeg_paths:
    if p.exists():
        tools_ffmpeg_bin = p.resolve()
        break

if tools_ffmpeg_bin:
    os.environ["PATH"] = str(tools_ffmpeg_bin) + os.pathsep + os.environ["PATH"]

from faster_whisper import WhisperModel
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoModel
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from agents.common.config import settings
from agents.common.database import get_db_session
from agents.common.models import Video, Transcript, VideoCaptions
from agents.common.enums import VisionModel

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use int8 for CPU to be faster/compatible, float16 for GPU
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        self.whisper_model_name = settings.WHISPER_MODEL
        self.whisper_model = None
        
        self.vision_model_name = settings.VISION_MODEL
        self.loaded_vision_model_name = None
        self.vision_processor = None
        self.vision_model = None
        
        # Ensure transcript folder exists
        settings.TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Transcription Agent initialized. Device: {self.device}, Compute: {self.compute_type}")

    def load_whisper_model(self, model_name=None):
        target_model = model_name or self.whisper_model_name
        
        # If already loaded and same model, return
        if self.whisper_model and self.whisper_model_name == target_model:
            return

        # If different model loaded, unload it
        if self.whisper_model:
            logger.info(f"Switching Whisper model from {self.whisper_model_name} to {target_model}...")
            del self.whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.whisper_model = None

        logger.info(f"Loading Faster-Whisper model: {target_model}...")
        try:
            self.whisper_model = WhisperModel(
                target_model, 
                device=self.device, 
                compute_type=self.compute_type
            )
            self.whisper_model_name = target_model
            logger.info("Whisper Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Whisper model: {e}")
            raise

    def load_vision_model(self, model_name=None):
        target_model = model_name or self.vision_model_name
        
        # If already loaded and same model, return
        if self.vision_model and self.loaded_vision_model_name == target_model:
            return

        # If different model loaded, unload it
        if self.vision_model:
            logger.info(f"Switching Vision model from {self.loaded_vision_model_name} to {target_model}...")
            del self.vision_model
            del self.vision_processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.vision_model = None

        logger.info(f"Loading Vision model: {target_model}...")
        try:
            if target_model == VisionModel.FLORENCE_2_LARGE:
                self.vision_processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)
                self.vision_model = AutoModelForCausalLM.from_pretrained(target_model, trust_remote_code=True).to(self.device)
            elif target_model == VisionModel.QWEN_2_5_VL:
                if process_vision_info is None:
                    raise ImportError("qwen-vl-utils is required for Qwen2.5-VL but not installed.")
                self.vision_processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)
                self.vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    target_model, 
                    torch_dtype="auto", 
                    device_map=self.device
                )
            elif target_model in [VisionModel.SIGLIP, VisionModel.CLIP]:
                # These are embedding models, not captioners. We load them to verify availability,
                # but process_visuals might need adjustment to use them effectively.
                # For now, we load them as AutoModel to prevent BLIP fallback failure.
                self.vision_processor = AutoProcessor.from_pretrained(target_model)
                self.vision_model = AutoModel.from_pretrained(target_model).to(self.device)
            else:
                # Default to BLIP
                self.vision_processor = BlipProcessor.from_pretrained(target_model)
                self.vision_model = BlipForConditionalGeneration.from_pretrained(target_model).to(self.device)
            
            self.loaded_vision_model_name = target_model
            self.vision_model_name = target_model
            logger.info("Vision Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Vision model: {e}")
            raise

    def process_visuals(self, video_path, model_name=None):
        """
        Extracts frames and generates captions for visual events.
        """
        self.load_vision_model(model_name)
        
        visual_segments = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 24 # Fallback
        
        interval_seconds = 5 # Analyze every 5 seconds
        frame_interval = int(fps * interval_seconds)
        frame_count = 0
        
        logger.info(f"Starting visual analysis for {video_path} using {self.vision_model_name}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    caption = ""
                    if self.vision_model_name == VisionModel.FLORENCE_2_LARGE:
                        # Florence-2 Generation
                        task_prompt = "<CAPTION>"
                        inputs = self.vision_processor(text=task_prompt, images=pil_image, return_tensors="pt").to(self.device)
                        
                        generated_ids = self.vision_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            early_stopping=False,
                            do_sample=False,
                            num_beams=3,
                        )
                        generated_text = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                        parsed_answer = self.vision_processor.post_process_generation(
                            generated_text, 
                            task=task_prompt, 
                            image_size=(pil_image.width, pil_image.height)
                        )
                        caption = parsed_answer.get(task_prompt, "")
                    elif self.vision_model_name == VisionModel.QWEN_2_5_VL:
                        # Qwen2.5-VL Generation
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": pil_image},
                                    {"type": "text", "text": "Describe this image concisely."},
                                ],
                            }
                        ]
                        text = self.vision_processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.vision_processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )
                        inputs = inputs.to(self.device)
                        
                        generated_ids = self.vision_model.generate(**inputs, max_new_tokens=128)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        caption = self.vision_processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                    else:
                        # BLIP Generation (Default)
                        # Note: SIGLIP/CLIP are not captioners, so they will likely fail here if selected.
                        # We fallback to BLIP logic which might error if model is not BLIP.
                        # Ideally we should block them or implement specific logic.
                        if self.vision_model_name in [VisionModel.SIGLIP, VisionModel.CLIP]:
                             caption = "Image embedding model selected. Captioning not supported."
                        else:
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

    def _sanitize_model_name(self, name):
        return name.replace("/", "_").replace("\\", "_")

    def process_video(self, video_id):
        # We fetch the video inside the method to ensure fresh state
        with get_db_session() as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video:
                logger.error(f"Video ID {video_id} not found in database.")
                return
            
            file_path = video.file_path
            filename = video.filename
            
        # Determine active models (Check for sidecar config)
        speech_model_name = self.whisper_model_name
        vision_model_name = self.vision_model_name
        
        sidecar_path = Path(file_path).with_suffix(".json")
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r") as f:
                    config = json.load(f)
                    if "whisper_model" in config:
                        speech_model_name = config["whisper_model"]
                        logger.info(f"Using sidecar Whisper model: {speech_model_name}")
                    if "vision_model" in config:
                        vision_model_name = config["vision_model"]
                        logger.info(f"Using sidecar Vision model: {vision_model_name}")
            except Exception as e:
                logger.warning(f"Failed to read sidecar config {sidecar_path}: {e}")
        
        # Check if artifacts exist
        transcript_exists = False
        captions_exist = False
        
        with get_db_session() as db:
            t = db.query(Transcript).filter(
                Transcript.video_id == video_id,
                Transcript.model_name == speech_model_name
            ).first()
            if t: transcript_exists = True
            
            c = db.query(VideoCaptions).filter(
                VideoCaptions.video_id == video_id,
                VideoCaptions.model_name == vision_model_name
            ).first()
            if c: captions_exist = True
            
        if transcript_exists and captions_exist:
            logger.info(f"Artifacts already exist for {filename} with models {speech_model_name}, {vision_model_name}. Skipping.")
            # Update status to pending_embedding if it was pending_transcription
            with get_db_session() as db:
                video = db.query(Video).filter(Video.id == video_id).first()
                if video and video.status == "pending_transcription":
                    video.status = "pending_embedding"
                    db.commit()
            return

        # Processing Block (No DB session active)
        try:
            base_name = Path(filename).stem
            
            # 1. Transcribe if needed
            if not transcript_exists:
                self.load_whisper_model(speech_model_name)
                logger.info(f"Transcribing {filename} with {speech_model_name}...")
                
                segments_generator, info = self.whisper_model.transcribe(file_path, beam_size=5)
                
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
                
                # Save Transcript Artifacts
                safe_model_name = self._sanitize_model_name(speech_model_name)
                json_path = settings.TRANSCRIPT_FOLDER / f"{base_name}_{safe_model_name}.json"
                vtt_path = settings.TRANSCRIPT_FOLDER / f"{base_name}_{safe_model_name}.vtt"
                
                result = {
                    "model": speech_model_name,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "segments": segments
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                
                self.save_vtt(segments, vtt_path)
                
                full_text = " ".join(full_text_parts)
                word_count = len(full_text.split())
                
                with get_db_session() as db:
                    new_transcript = Transcript(
                        video_id=video_id,
                        model_name=speech_model_name,
                        full_text=full_text,
                        vtt_file_path=str(vtt_path),
                        json_file_path=str(json_path),
                        word_count=word_count
                    )
                    db.add(new_transcript)
                    db.commit()
                    logger.info(f"Saved transcript for {filename} ({speech_model_name}).")

            # 2. Visual Analysis if needed
            if not captions_exist:
                logger.info(f"Generating captions for {filename} with {vision_model_name}...")
                visual_segments = self.process_visuals(file_path, vision_model_name)
                
                # Save Caption Artifacts
                safe_model_name = self._sanitize_model_name(vision_model_name)
                json_path = settings.TRANSCRIPT_FOLDER / f"{base_name}_{safe_model_name}_captions.json"
                
                result = {
                    "model": vision_model_name,
                    "segments": visual_segments
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                
                full_text = " ".join([s["text"] for s in visual_segments])
                
                with get_db_session() as db:
                    new_captions = VideoCaptions(
                        video_id=video_id,
                        model_name=vision_model_name,
                        full_text=full_text,
                        json_file_path=str(json_path)
                    )
                    db.add(new_captions)
                    db.commit()
                    logger.info(f"Saved captions for {filename} ({vision_model_name}).")

            # Update Video Status
            with get_db_session() as db:
                video = db.query(Video).filter(Video.id == video_id).first()
                if video:
                    video.status = "pending_embedding"
                    db.commit()
                    logger.info(f"Completed processing for {filename}. Status updated.")

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            # Optional: Mark as error in DB to avoid infinite retries
            # with get_db_session() as db:
            #     video = db.query(Video).filter(Video.id == video_id).first()
            #     video.status = "error_transcription"
            #     db.commit()

    def check_and_queue_missing_artifacts(self, speech_model, vision_model):
        """
        Iterates through all videos and queues them for processing if artifacts
        for the specified models are missing.
        """
        with get_db_session() as db:
            videos = db.query(Video).all()
            count = 0
            for video in videos:
                # Check Transcript
                t = db.query(Transcript).filter(
                    Transcript.video_id == video.id,
                    Transcript.model_name == speech_model
                ).first()
                
                # Check Captions
                c = db.query(VideoCaptions).filter(
                    VideoCaptions.video_id == video.id,
                    VideoCaptions.model_name == vision_model
                ).first()
                
                if not t or not c:
                    # Only queue if not already processing or queued
                    if video.status not in ["pending_transcription", "processing"]:
                        video.status = "pending_transcription"
                        count += 1
            
            if count > 0:
                db.commit()
                logger.info(f"Queued {count} videos for processing with models {speech_model}, {vision_model}.")
            else:
                logger.info("All videos have artifacts for the specified models.")

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