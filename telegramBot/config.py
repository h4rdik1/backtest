# config.py — All constants and environment variables

import os
import asyncio
from dotenv import load_dotenv

# Fix for Windows asyncio bug
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load secrets from .env file
load_dotenv()

TOKEN          = os.getenv("TELEGRAM_BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Folder where chart images are saved
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
