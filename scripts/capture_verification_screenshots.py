import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    index_dir = root / "index_store"
    results_dir = root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup sample notes
    sample_src = data_dir / "sample_ml_notes.txt"
    backup_sample = root / "scripts" / "sample_ml_notes.txt"
    if sample_src.exists():
        shutil.copy2(sample_src, backup_sample)

    # 1. Clean data and index_store for empty welcome state
    for f in data_dir.glob("*"):
        if f.is_file():
            f.unlink()
    for f in index_dir.glob("*"):
        if f.is_file():
            f.unlink()

    env = os.environ.copy()
    env["BUBOT_BACKEND"] = "lite"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    port = 8540
    print(f"Starting Streamlit on port {port}...", flush=True)
    proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ], env=env)

    try:
        time.sleep(6)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            url = f"http://127.0.0.1:{port}"
            print(f"Navigating to {url}...", flush=True)
            page.goto(url, timeout=30000)
            page.wait_for_selector('h1', timeout=25000)
            time.sleep(2)

            # Screenshot A: Empty welcome state
            welcome_png = results_dir / "screenshot_a_welcome.png"
            page.screenshot(path=str(welcome_png))
            print(f"Screenshot (a) saved to {welcome_png}", flush=True)

            # Step B: Upload file via file input directly (works on hidden file inputs)
            print("Uploading sample_ml_notes.txt...", flush=True)
            page.set_input_files('input[type="file"]', str(backup_sample))
            
            # Wait for reload into chat state (model load + index takes ~35s)
            print("Waiting for chat state header...", flush=True)
            page.wait_for_selector('h2', timeout=70000)
            time.sleep(2)

            # Submit question: "What is backpropagation?"
            print("Entering chat question...", flush=True)
            chat_input = page.wait_for_selector('[data-testid="stChatInput"] textarea', timeout=15000)
            chat_input.fill("What is backpropagation?")
            chat_input.press("Enter")

            # Wait for answer
            print("Waiting for assistant answer...", flush=True)
            page.wait_for_selector('[data-testid="stChatMessage"]:has([aria-label*="assistant"])', timeout=40000)
            time.sleep(3)

            # Screenshot B: Chat state
            chat_png = results_dir / "screenshot_b_chat.png"
            page.screenshot(path=str(chat_png))
            print(f"Screenshot (b) saved to {chat_png}", flush=True)

            browser.close()
    finally:
        proc.terminate()
        proc.wait()
        # Restore sample file in data dir
        if backup_sample.exists():
            shutil.copy2(backup_sample, data_dir / "sample_ml_notes.txt")

if __name__ == "__main__":
    main()
