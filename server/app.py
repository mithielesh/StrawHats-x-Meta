import sys
import os
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    """Entry point for the OpenEnv multi-mode deployment."""
    # This just tells it to run your existing main.py app
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()