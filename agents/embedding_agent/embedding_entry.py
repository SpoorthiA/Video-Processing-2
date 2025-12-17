import time
import logging
import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from agents.common.config import settings
from agents.embedding_agent.embedding_processor import EmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("embedding_agent")

def main():
    logger.info("Starting Embedding & Indexing Agent...")
    
    try:
        processor = EmbeddingProcessor()
        # Using Ingestion interval as a default or we could add a new setting
        poll_interval = 10 
        
        while True:
            processed_count = processor.run_once()
            if processed_count == 0:
                time.sleep(poll_interval)
            else:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Embedding Agent stopped by user.")
    except Exception as e:
        logger.critical(f"Embedding Agent crashed: {e}", exc_info=True)

if __name__ == "__main__":
    main()