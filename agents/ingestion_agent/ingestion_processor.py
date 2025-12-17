import shutil
import logging
import ffmpeg
from pathlib import Path
from sqlalchemy.exc import IntegrityError
from agents.common.config import settings
from agents.common.database import get_db_session
from agents.common.models import Video

logger = logging.getLogger(__name__)

class IngestionAgent:
    def __init__(self):
        self.inbox_dir = settings.VIDEO_INBOX_FOLDER
        self.processed_dir = settings.VIDEO_PROCESSED_FOLDER
        
        # Ensure directories exist
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_video_metadata(self, file_path: Path):
        """Extracts metadata using ffprobe."""
        try:
            probe = ffmpeg.probe(str(file_path))
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                raise ValueError("No video stream found")

            duration = float(probe['format'].get('duration', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            fps_eval = video_stream.get('r_frame_rate', '0/0').split('/')
            fps = float(fps_eval[0]) / float(fps_eval[1]) if len(fps_eval) == 2 and float(fps_eval[1]) > 0 else 0
            
            return {
                "duration": duration,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "size": file_path.stat().st_size
            }
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error processing {file_path}: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error extracting metadata for {file_path}: {e}")
            return None

    def process_file(self, file_path: Path):
        """Ingests a single video file."""
        logger.info(f"Processing new file: {file_path.name}")
        
        metadata = self.get_video_metadata(file_path)
        if not metadata:
            logger.warning(f"Skipping {file_path.name} due to metadata extraction failure.")
            return

        # Database Transaction
        with get_db_session() as db:
            try:
                # Check if file already exists (by filename)
                existing_video = db.query(Video).filter(Video.filename == file_path.name).first()
                if existing_video:
                    logger.warning(f"Video '{file_path.name}' already exists in DB. Moving to processed.")
                    # Move logic below will handle the file cleanup
                else:
                    new_video = Video(
                        filename=file_path.name,
                        file_path=str(self.processed_dir / file_path.name), # New path
                        duration_seconds=metadata["duration"],
                        file_size_bytes=metadata["size"],
                        fps=metadata["fps"],
                        resolution=metadata["resolution"],
                        status="pending_transcription"
                    )
                    db.add(new_video)
                    db.commit()
                    logger.info(f"Registered '{file_path.name}' in database.")
                
                # Move the file to the processed directory
                destination = self.processed_dir / file_path.name
                shutil.move(str(file_path), str(destination))
                logger.info(f"Moved file to {destination}")

            except IntegrityError:
                db.rollback()
                logger.error(f"Database integrity error for {file_path.name}")
            except Exception as e:
                db.rollback()
                logger.error(f"Error ingesting {file_path.name}: {e}", exc_info=True)

    def run_once(self):
        """Scans the inbox and processes all found videos."""
        # Supported extensions
        extensions = ['*.mp4', '*.mov', '*.mkv', '*.avi']
        files = []
        for ext in extensions:
            files.extend(self.inbox_dir.glob(ext))
        
        if not files:
            return 0

        logger.info(f"Found {len(files)} videos in inbox.")
        for file_path in files:
            self.process_file(file_path)
        
        return len(files)