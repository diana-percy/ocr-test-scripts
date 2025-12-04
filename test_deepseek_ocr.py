#!/usr/bin/env python3
"""
Test DeepSeek-OCR text extraction from images and PDFs via chat completions endpoint

Usage: python test_deepseek_ocr.py <filename>
Example: python test_deepseek_ocr.py receipt.jpeg
         python test_deepseek_ocr.py document.pdf

This script uses the /v1/chat/completions endpoint with multimodal messages.

For PDFs, pages are converted to images first (DeepSeek-OCR doesn't accept PDFs directly).

Prompts used:
- All inputs: "Free OCR." (grounding prompts from GitHub don't work on Vertex AI)

Input files are automatically read from the 'resources' folder.
"""

import json
import sys

from openai import OpenAI
from config import API_KEY, BASE_URL
from utils import (
    resolve_input_path, RESOURCES_DIR, get_file_type, get_mime_type, 
    encode_file, pdf_to_images, encode_image_bytes
)

# DeepSeek-OCR model name (as configured in LiteLLM)
DEEPSEEK_OCR_MODEL = "deepseek-ocr"

# DeepSeek OCR prompts based on input type
# Note: Vertex AI hosted version has issues with grounding prompts, so we use "Free OCR." for all inputs
# The grounding prompts from the GitHub repo don't work reliably on Vertex AI
PROMPTS = {
    "document": "Free OCR.",  # For PDFs/documents
    "image": "Free OCR.",  # For standalone images
}


def ocr_single_image(client, base64_data, mime_type, prompt):
    """Run OCR on a single image and return the extracted text"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=DEEPSEEK_OCR_MODEL,
        messages=messages,
        max_tokens=4096,
        temperature=0.0,
    )
    
    return response


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_deepseek_ocr.py <filename>")
        print("Example: python test_deepseek_ocr.py receipt.jpeg")
        print("         python test_deepseek_ocr.py document.pdf")
        print(f"\nInput files are read from: {RESOURCES_DIR}")
        sys.exit(1)

    input_filename = sys.argv[1]
    input_path = resolve_input_path(input_filename)

    # Determine file type
    try:
        file_type = get_file_type(input_path)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Select prompt based on file type
    if file_type == 'pdf':
        prompt = PROMPTS["document"]
    else:
        prompt = PROMPTS["image"]

    # Create OpenAI client pointing to LiteLLM proxy
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    if file_type == 'image':
        # Direct image processing
        print(f"📸 Encoding image: {input_path}")
        try:
            base64_data = encode_file(input_path)
            mime_type = get_mime_type(input_path)
            print(f"✅ Image encoded ({len(base64_data)} characters)")
            print(f"   MIME type: {mime_type}")
        except FileNotFoundError:
            print(f"❌ Error: File not found: {input_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error encoding file: {e}")
            sys.exit(1)

        print(f"\n🚀 Sending to DeepSeek-OCR model: {DEEPSEEK_OCR_MODEL}")
        print("   Using /v1/chat/completions endpoint")
        print(f"   Prompt: {prompt}")

        try:
            response = ocr_single_image(client, base64_data, mime_type, prompt)

            print("\n" + "="*60)
            print("RAW API RESPONSE")
            print("="*60)
            print(json.dumps(response.model_dump(), indent=2, default=str))

            print("\n" + "="*60)
            print("EXTRACTED TEXT")
            print("="*60)
            print(response.choices[0].message.content)

        except Exception as e:
            print(f"\n❌ Error calling API: {e}")
            sys.exit(1)

    else:
        # PDF processing - convert to images first
        print(f"📄 Converting PDF to images: {input_path}")
        try:
            page_images = pdf_to_images(input_path)
            print(f"✅ PDF converted to {len(page_images)} page image(s)")
        except ImportError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"❌ Error: File not found: {input_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error converting PDF: {e}")
            sys.exit(1)

        print(f"\n🚀 Sending to DeepSeek-OCR model: {DEEPSEEK_OCR_MODEL}")
        print("   Using /v1/chat/completions endpoint")
        print(f"   Prompt: {prompt}")
        print(f"   Processing {len(page_images)} page(s)...")

        all_text = []
        
        for page_num, (img_bytes, mime_type) in enumerate(page_images, 1):
            print(f"\n   📄 Processing page {page_num}/{len(page_images)}...")
            
            try:
                base64_data = encode_image_bytes(img_bytes)
                response = ocr_single_image(client, base64_data, mime_type, prompt)
                page_text = response.choices[0].message.content
                all_text.append(f"--- Page {page_num} ---\n{page_text}")
                print(f"   ✅ Page {page_num} complete ({len(page_text)} characters)")
            except Exception as e:
                print(f"   ❌ Error on page {page_num}: {e}")
                all_text.append(f"--- Page {page_num} ---\n[Error: {e}]")

        print("\n" + "="*60)
        print("EXTRACTED TEXT (ALL PAGES)")
        print("="*60)
        print("\n\n".join(all_text))


if __name__ == "__main__":
    main()
