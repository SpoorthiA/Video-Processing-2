import multiprocessing
import time
import sys
import os
import socket
import uvicorn

# Ensure the current directory is in the path so imports work
sys.path.append(os.getcwd())

from agents.common.config import settings

def find_available_port(start_port, max_tries=10):
    """
    Finds an available port starting from start_port.
    Returns the first available port or raises an exception if none found.
    """
    for port in range(start_port, start_port + max_tries):
        # Try to bind to the port to check availability
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Bind to the specific host and port
                s.bind((settings.API_HOST, port))
                # If successful, the port is free. The socket is closed when exiting the 'with' block.
                return port
            except OSError:
                # Port is in use, try the next one
                continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_tries}")

def start_api(port):
    """Runs the FastAPI server on the specified port."""
    print(f"Starting API on port {port}...")
    # Using a string import for uvicorn is safer for multiprocessing
    uvicorn.run("agents.api.api_service:app", host=settings.API_HOST, port=port, reload=False)

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

    # Find an available port for the API
    try:
        api_port = find_available_port(settings.API_PORT)
        print(f"Selected available port for API: {api_port}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create processes
    # Pass the discovered port to the API process
    p_api = multiprocessing.Process(target=start_api, args=(api_port,), name="API")
    p_ingest = multiprocessing.Process(target=start_ingestion, name="Ingestion")
    p_transcribe = multiprocessing.Process(target=start_transcription, name="Transcription")
    p_embed = multiprocessing.Process(target=start_embedding, name="Embedding")

    processes = [p_api, p_ingest, p_transcribe, p_embed]

    try:
        # Start all processes
        for p in processes:
            p.start()
            time.sleep(1) # Stagger start slightly

        print("\n" + "="*50)
        print(f"🚀 System is running!")
        print(f"🌐 Open the UI at: http://localhost:{api_port}")
        print("="*50 + "\n")
        print("Press Ctrl+C to stop.")
        
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