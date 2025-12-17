import os
from agents.common.database import engine, Base
from agents.common.config import settings
import agents.common.models  # Import models to register them with Base
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Initializes the SQLite database.
    This function creates the database file and all necessary tables
    as defined by the SQLAlchemy models.
    """
    
    # The DATABASE_URL for sqlite is 'sqlite:///./video_retrieval_poc.db'
    # We need to extract the file path from this URL.
    db_path_str = settings.DATABASE_URL.split('///')[1]
    db_path = os.path.abspath(db_path_str)

    # Check if the database file already exists to avoid overwriting it.
    # if os.path.exists(db_path):
    #     logging.warning(f"Database file already exists at '{db_path}'.")
    #     logging.info("To re-initialize the database, please delete this file first.")
    #     return

    logging.info(f"Ensuring database tables exist at '{db_path}'...")
    
    try:
        # This is the command that creates all tables defined in models.py
        Base.metadata.create_all(bind=engine)
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"An error occurred during database initialization: {e}", exc_info=True)


if __name__ == "__main__":
    main()