# Entry point for Hugging Face Spaces
# HF Spaces requires app.py at the root level.
# This file simply redirects to the actual Streamlit app in frontend/app.py

import subprocess
import sys
import os

# Run the actual app
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
     "--server.port", "7860",
     "--server.address", "0.0.0.0"],
    check=True
)
