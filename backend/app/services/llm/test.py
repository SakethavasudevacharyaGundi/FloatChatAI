# test_env.py

from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(".env")

load_dotenv(env_path)

print(
    os.getenv("GEMINI_KEY_1")
)