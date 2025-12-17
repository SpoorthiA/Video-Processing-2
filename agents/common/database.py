from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from .config import settings

# Create the SQLAlchemy engine using the database URL from settings
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True # Helps prevent connection errors
)

# Create a configured "Session" class. This is the factory for all new sessions.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for our SQLAlchemy ORM models. All models will inherit from this.
Base = declarative_base()

@contextmanager
def get_db_session():
    """
    Provides a new database session for a single unit of work.
    This function is designed to be used in a 'with' statement
    or context manager to ensure the session is always closed.
    
    Example:
    with get_db_session() as db:
        db.query(...)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()