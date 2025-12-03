"""
Configuration for OCR test scripts

Create a .env file with your API_KEY and BASE_URL values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

OCR_MODEL = "mistral-ocr-2505"

