import os
import sys
import subprocess
import time
import urllib.request
from playwright.sync_api import sync_playwright

def main():
    env = os.environ.copy()
    env["BUBOT_BACKEND"] = "lite"
    env["PYTHONUNBUFFERED"] = "1"
    
    port = 8520
    print(f"Starting Streamlit on port {port}...", flush=True)
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to be responsive
    started = False
    for i in range(25):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as resp:
                if resp.status == 200:
                    print("Streamlit server is healthy!", flush=True)
                    started = True
                    break
        except Exception:
            pass
        if proc.poll() is not None:
            out, _ = proc.communicate()
            print("Streamlit process died prematurely:", out, flush=True)
            return

    if not started:
        print("Streamlit did not become healthy in time.", flush=True)
        proc.terminate()
        return

    try:
        with sync_playwright() as p:
            print("Launching Chromium...", flush=True)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            print(f"Navigating to http://127.0.0.1:{port}...", flush=True)
            page.goto(f"http://127.0.0.1:{port}", timeout=30000)
            page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=20000)
            time.sleep(2)
            os.makedirs("eval/results", exist_ok=True)
            welcome_path = "eval/results/screenshot_a_welcome.png"
            page.screenshot(path=welcome_path)
            print(f"Screenshot (a) saved to {welcome_path}", flush=True)
            browser.close()
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
