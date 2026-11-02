"""
HuggingFace Spaces Docker entry point.
Starts FastAPI in background thread, then Streamlit on port 7860.
"""
import subprocess
import threading
import time
import sys
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"


def start_fastapi():
    import uvicorn
    print("[API] Starting FastAPI on port 8000...")
    try:
        uvicorn.run(
            "api.rag_api:app",
            host="0.0.0.0",
            port=8000,
            log_level="warning"
        )
    except Exception as e:
        print(f"[API] Error: {e}")


def wait_for_api(max_retries=40, delay=3):
    import requests
    print("[API] Waiting for FastAPI...")
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.ok:
                print(f"[API] Ready after {attempt} attempts")
                return True
        except Exception:
            pass
        print(f"[API] Attempt {attempt}/{max_retries}...")
        time.sleep(delay)
    print("[API] WARNING: Starting Streamlit anyway")
    return False


if __name__ == "__main__":
    print("=" * 50)
    print("Enterprise RAG Platform — Starting")
    print("=" * 50)

    api_thread = threading.Thread(target=start_fastapi, daemon=True)
    api_thread.start()

    wait_for_api()

    print("[UI] Starting Streamlit on port 7860...")
    result = subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--browser.gatherUsageStats=false",
        "--logger.level=warning"
    ])
    sys.exit(result.returncode)