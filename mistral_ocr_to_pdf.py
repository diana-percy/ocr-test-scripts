#!/usr/bin/env python3
"""
Convert images or PDFs to searchable/OCR-processed PDFs using Mistral OCR via LiteLLM

Usage: python mistral_ocr_to_pdf.py <filename> [output.pdf]
Example: python mistral_ocr_to_pdf.py scan.jpg
         python mistral_ocr_to_pdf.py document.pdf extracted_document.pdf

Input files are automatically read from the 'resources' folder.
Output files are automatically saved to the 'output' folder.

Supported input formats:
- Images: jpg, jpeg, png, gif, webp
- Documents: pdf

This script:
1. Sends the image/PDF to Mistral OCR for text extraction via /v1/ocr endpoint
2. Outputs the raw markdown text to a PDF with embedded images

Required packages:
    pip install requests reportlab pillow
"""

import requests
import sys
import os

from config import API_KEY, BASE_URL
from utils import (
    resolve_input_path, resolve_output_path, RESOURCES_DIR, OUTPUT_DIR,
    get_file_type, get_mime_type, encode_file,
    clean_text_for_preview, markdown_to_pdf
)

# Mistral OCR model name
MISTRAL_OCR_MODEL = "mistral-ocr-2505"


def extract_text_with_mistral_ocr(file_path):
    """Extract text and images from image or PDF using Mistral OCR"""
    file_type = get_file_type(file_path)
    mime_type = get_mime_type(file_path)
    
    if file_type == 'image':
        print(f"📸 Encoding image: {file_path}")
    else:
        print(f"📄 Encoding PDF: {file_path}")
    
    base64_data = encode_file(file_path)
    print(f"✅ File encoded ({len(base64_data)} characters)")

    print(f"🚀 Sending to Mistral OCR model: {MISTRAL_OCR_MODEL}")
    
    # Build the document payload based on file type
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
    
    response = requests.post(
        f"{BASE_URL}/v1/ocr",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MISTRAL_OCR_MODEL,
            "document": document,
            "include_image_base64": True
        }
    )
    
    response.raise_for_status()
    result = response.json()
    
    # Extract text from all pages
    all_text = []
    all_images = {}  # Map image IDs to base64 data
    
    if "pages" in result:
        for page_idx, page in enumerate(result["pages"]):
            text = page.get("markdown") or page.get("text") or page.get("content") or ""
            if text:
                all_text.append(text)
            
            # Extract images with their base64 data
            for img in page.get("images", []):
                img_id = img.get("id")
                img_base64 = img.get("image_base64")
                if img_id and img_base64:
                    all_images[img_id] = img_base64
    
    return "\n\n".join(all_text), all_images, result.get("usage_info", {})


def main():
    if len(sys.argv) < 2:
        print("Usage: python mistral_ocr_to_pdf.py <filename> [output.pdf]")
        print("Example: python mistral_ocr_to_pdf.py scan.jpg")
        print("         python mistral_ocr_to_pdf.py document.pdf extracted_document.pdf")
        print("\nSupported formats: jpg, jpeg, png, gif, webp, pdf")
        print(f"\nInput files are read from: {RESOURCES_DIR}")
        print(f"Output files are saved to: {OUTPUT_DIR}")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    input_path = resolve_input_path(input_filename)
    
    # Determine output path
    if len(sys.argv) >= 3:
        output_path = resolve_output_path(sys.argv[2])
    else:
        # Default: same name as input but with _ocr.pdf extension in output folder
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = resolve_output_path(f"{base_name}_ocr.pdf")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        print(f"   (Also checked: {os.path.join(RESOURCES_DIR, input_filename)})")
        sys.exit(1)
    
    try:
        # Validate file type
        file_type = get_file_type(input_path)
        
        # Step 1: Extract text using OCR
        print("\n" + "="*60)
        print(f"STEP 1: Extracting text and images with Mistral OCR ({file_type.upper()})")
        print("="*60)
        
        markdown_text, images, usage_info = extract_text_with_mistral_ocr(input_path)
        
        if not markdown_text.strip():
            print("⚠️  Warning: No text was extracted from the file")
            sys.exit(1)
        
        print(f"✅ Extracted {len(markdown_text)} characters of text")
        print(f"✅ Found {len(images)} embedded images")
        
        # Show preview
        print("\n📝 Text preview:")
        preview = clean_text_for_preview(markdown_text)[:500]
        print("-" * 40)
        print(preview + ("..." if len(markdown_text) > 500 else ""))
        print("-" * 40)
        
        # Step 2: Convert to PDF
        print("\n" + "="*60)
        print("STEP 2: Generating PDF")
        print("="*60)
        
        markdown_to_pdf(markdown_text, output_path, images, model_name=MISTRAL_OCR_MODEL)
        
        print(f"✅ PDF saved to: {output_path}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"   Input file:   {input_path} ({file_type})")
        print(f"   Output PDF:   {output_path}")
        print(f"   Text length:  {len(markdown_text)} characters")
        print(f"   Images:       {len(images)}")
        if usage_info:
            print(f"   OCR usage:    {usage_info}")
        
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ OCR API Error: {e}")
        print(f"   Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

