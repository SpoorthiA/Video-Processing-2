import time
import logging
import sys
import shutil
import os
from pathlib import Path
from agents.common.config import settings
from agents.ingestion_agent.ingestion_processor import IngestionAgent

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestion_agent")

# Setup FFmpeg Path
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
    logger.info(f"FFmpeg found at: {tools_ffmpeg_bin}")

def check_dependencies():
    """Verifies that necessary external tools are installed."""
    if not shutil.which("ffmpeg"):
        logger.critical("Error: 'ffmpeg' is not found in the system PATH.")
        logger.critical("Please install FFmpeg and add it to your PATH to continue.")
        return False
    
    if not shutil.which("ffprobe"):
        logger.critical("Error: 'ffprobe' is not found in the system PATH.")
        logger.critical("It is usually installed alongside ffmpeg.")
        return False
        
    return True

def main():
    if not check_dependencies():
        sys.exit(1)
        
    logger.info("Starting Ingestion Agent...")
    agent = IngestionAgent()
    
    try:
        while True:
            processed_count = agent.run_once()
            if processed_count == 0:
                # No files found, sleep for the configured interval
                time.sleep(settings.INGESTION_AGENT_INTERVAL)
            else:
                # If we processed files, check again immediately in case more arrived
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Ingestion Agent stopped by user.")
    except Exception as e:
        logger.critical(f"Ingestion Agent crashed: {e}", exc_info=True)

if __name__ == "__main__":
    main()