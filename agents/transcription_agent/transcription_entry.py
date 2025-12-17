import time
import logging
import sys
import os

# Ensure project root is in path if run directly (though -m is preferred)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from agents.common.config import settings
from agents.transcription_agent.transcription_processor import TranscriptionProcessor

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("transcription_agent")

def main():
    logger.info("Starting Transcription Agent...")
    
    try:
        processor = TranscriptionProcessor()
        logger.info(f"Polling for videos every {settings.TRANSCRIPTION_AGENT_INTERVAL} seconds.")
        
        while True:
            processed_count = processor.run_once()
            if processed_count == 0:
                time.sleep(settings.TRANSCRIPTION_AGENT_INTERVAL)
            else:
                # If we processed something, check immediately for more
                time.sleep(1) 
                
    except KeyboardInterrupt:
        logger.info("Stopping Transcription Agent (User Interrupt)...")
    except Exception as e:
        logger.critical(f"Transcription Agent crashed: {e}", exc_info=True)

if __name__ == "__main__":
    main()