#!/usr/bin/env python3
"""
Test OCR text extraction from images via LiteLLM

Usage: python test_ocr_image.py <filename>
Example: python test_ocr_image.py receipt.jpeg

Input files are automatically read from the 'resources' folder.

Configure the OCR model and API settings in config.py
"""

import requests
import base64
import sys
import os

from config import API_KEY, BASE_URL, OCR_MODEL
from utils import resolve_input_path, RESOURCES_DIR


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main():
    # Check if image path was provided
    if len(sys.argv) < 2:
        print("Usage: python test_ocr_image.py <filename>")
        print("Example: python test_ocr_image.py receipt.jpeg")
        print(f"\nInput files are read from: {RESOURCES_DIR}")
        sys.exit(1)

    image_filename = sys.argv[1]
    image_path = resolve_input_path(image_filename)

    print(f"📸 Encoding image: {image_path}")
    try:
        base64_image = encode_image(image_path)
        print(f"✅ Image encoded ({len(base64_image)} characters)")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {image_path}")
        print(f"   (Also checked: {os.path.join(RESOURCES_DIR, image_filename)})")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        sys.exit(1)

    print(f"\n🚀 Sending to OCR model: {OCR_MODEL}")

    try:
        # Use the /ocr endpoint with image_url type for images
        response = requests.post(
            f"{BASE_URL}/v1/ocr",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OCR_MODEL,
                "document": {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                },
                "include_image_base64": True
            }
        )
        
        response.raise_for_status()
        result = response.json()

        # Print raw API response
        import json
        print("\n" + "="*60)
        print("RAW API RESPONSE")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"\n❌ Error calling API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

