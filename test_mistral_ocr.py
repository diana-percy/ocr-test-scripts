#!/usr/bin/env python3
"""
Test Mistral OCR text extraction from images and PDFs via LiteLLM /ocr endpoint

Usage: python test_mistral_ocr.py <filename> [--include-images]
Example: python test_mistral_ocr.py receipt.jpeg
         python test_mistral_ocr.py document.pdf
         python test_mistral_ocr.py document.pdf --include-images

Input files are automatically read from the 'resources' folder.

This script uses the /v1/ocr endpoint which is specifically supported
for Mistral OCR models.
"""

import requests
import base64
import sys
import os
import argparse
import json

from config import API_KEY, BASE_URL
from utils import resolve_input_path, RESOURCES_DIR, get_file_type, get_mime_type

# Mistral OCR model name
MISTRAL_OCR_MODEL = "mistral-ocr-2505"


def encode_file(file_path):
    """Encode file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description="Test Mistral OCR text extraction from images and PDFs")
    parser.add_argument("filename", help="Image or PDF file to process")
    parser.add_argument("--include-images", action="store_true", 
                        help="Include base64 image embeddings in API response")
    args = parser.parse_args()

    input_filename = args.filename
    input_path = resolve_input_path(input_filename)

    # Determine file type
    try:
        file_type = get_file_type(input_path)
        mime_type = get_mime_type(input_path)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    if file_type == 'image':
        print(f"📸 Encoding image: {input_path}")
    else:
        print(f"📄 Encoding PDF: {input_path}")

    try:
        base64_data = encode_file(input_path)
        print(f"✅ File encoded ({len(base64_data)} characters)")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_path}")
        print(f"   (Also checked: {os.path.join(RESOURCES_DIR, input_filename)})")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error encoding file: {e}")
        sys.exit(1)

    print(f"\n🚀 Sending to Mistral OCR model: {MISTRAL_OCR_MODEL}")
    print("   Using /v1/ocr endpoint")
    if file_type == 'pdf':
        print("⚠️  Note: PDFs can take longer to process")

    # Build document payload based on file type
    if file_type == 'image':
        document = {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{base64_data}"
        }
    else:  # PDF
        document = {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{base64_data}"
        }

    try:
        response = requests.post(
            f"{BASE_URL}/v1/ocr",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MISTRAL_OCR_MODEL,
                "document": document,
                "include_image_base64": args.include_images
            }
        )
        
        response.raise_for_status()
        result = response.json()

        print("\n" + "="*60)
        print("RAW API RESPONSE")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"\n❌ Error calling API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

