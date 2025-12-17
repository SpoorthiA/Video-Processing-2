import multiprocessing
import time
import sys
import os
import uvicorn

# Ensure the current directory is in the path so imports work
sys.path.append(os.getcwd())

from agents.common.config import settings

def start_api():
    """Runs the FastAPI server."""
    print("Starting API...")
    # Using a string import for uvicorn is safer for multiprocessing
    uvicorn.run("agents.api.api_service:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)

def start_ingestion():
    """Runs the Ingestion Agent."""
    print("Starting Ingestion Agent...")
    from agents.ingestion_agent.ingestion_entry import main
    main()

def start_transcription():
    """Runs the Transcription Agent."""
    print("Starting Transcription Agent...")
    from agents.transcription_agent.transcription_entry import main
    main()

def start_embedding():
    """Runs the Embedding Agent."""
    print("Starting Embedding Agent...")
    from agents.embedding_agent.embedding_entry import main
    main()

if __name__ == "__main__":
    # Force multiprocessing to use the current python executable (venv)
    # This prevents subprocesses from spawning with the system python
    if sys.platform == 'win32':
        multiprocessing.set_executable(sys.executable)

    print(f"Running with Python: {sys.executable}")

    # Initialize Database if needed
    print("Checking database...")
    try:
        import setup_database
        # We need to force the creation if it doesn't exist, setup_database.main() handles the check
        setup_database.main()
    except Exception as e:
        print(f"Database initialization failed: {e}")
        sys.exit(1)

    # Create processes
    p_api = multiprocessing.Process(target=start_api, name="API")
    p_ingest = multiprocessing.Process(target=start_ingestion, name="Ingestion")
    p_transcribe = multiprocessing.Process(target=start_transcription, name="Transcription")
    p_embed = multiprocessing.Process(target=start_embedding, name="Embedding")

    processes = [p_api, p_ingest, p_transcribe, p_embed]

    try:
        # Start all processes
        for p in processes:
            p.start()
            time.sleep(1) # Stagger start slightly

        print("All agents are running. Press Ctrl+C to stop.")
        
        # Monitor processes
        while True:
            time.sleep(1)
            # Check if any process has died unexpectedly
            for p in processes:
                if not p.is_alive():
                    # If a process dies, we could restart it, but for now we just log it
                    # print(f"Process {p.name} is not running.")
                    pass
                    
    except KeyboardInterrupt:
        print("\nStopping all agents...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        print("System stopped.")