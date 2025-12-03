#!/usr/bin/env python3
"""
Test OCR text extraction from PDF files via LiteLLM

Usage: python test_ocr_pdf.py <filename> [--include-images]
Example: python test_ocr_pdf.py document.pdf
         python test_ocr_pdf.py document.pdf --include-images

Input files are automatically read from the 'resources' folder.

Configure the OCR model and API settings in config.py
"""

import requests
import base64
import sys
import os
import argparse

from config import API_KEY, BASE_URL, OCR_MODEL
from utils import resolve_input_path, RESOURCES_DIR


def encode_pdf(pdf_path):
    """Encode PDF to base64"""
    with open(pdf_path, "rb") as pdf_file:
        return base64.b64encode(pdf_file.read()).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(description="Test OCR text extraction from PDF files")
    parser.add_argument("filename", help="PDF file to process")
    parser.add_argument("--include-images", action="store_true", 
                        help="Include base64 image embeddings in API response")
    args = parser.parse_args()

    pdf_filename = args.filename
    pdf_path = resolve_input_path(pdf_filename)

    print(f"📄 Encoding PDF: {pdf_path}")
    try:
        base64_pdf = encode_pdf(pdf_path)
        print(f"✅ PDF encoded ({len(base64_pdf)} characters)")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {pdf_path}")
        print(f"   (Also checked: {os.path.join(RESOURCES_DIR, pdf_filename)})")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error encoding PDF: {e}")
        sys.exit(1)

    print(f"\n🚀 Sending to OCR model: {OCR_MODEL}")
    print("⚠️  Note: PDFs can take longer to process")

    try:
        # Use the /ocr endpoint with document_url type for PDFs
        response = requests.post(
            f"{BASE_URL}/v1/ocr",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OCR_MODEL,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                "include_image_base64": args.include_images
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

