#!/usr/bin/env python3
"""
Convert images or PDFs to searchable/OCR-processed PDFs using DeepSeek OCR via LiteLLM

Usage: python deepseek_ocr_to_pdf.py <filename> [output.pdf]
Example: python deepseek_ocr_to_pdf.py scan.jpg
         python deepseek_ocr_to_pdf.py document.pdf extracted_document.pdf

Input files are automatically read from the 'resources' folder.
Output files are automatically saved to the 'output' folder.

Supported input formats:
- Images: jpg, jpeg, png, gif, webp
- Documents: pdf (converted to images first, then OCR'd page by page)

This script:
1. Sends the image/PDF to DeepSeek OCR for text extraction via /v1/chat/completions endpoint
2. For PDFs: Uses grounding prompts to extract embedded images
3. Outputs the extracted text and images to a PDF

Required packages:
    pip install openai reportlab pillow pymupdf
"""

import sys
import os
import re
import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image as PILImage
from config import API_KEY, BASE_URL
from utils import (
    resolve_input_path, resolve_output_path, RESOURCES_DIR, OUTPUT_DIR,
    get_file_type, get_mime_type, encode_file,
    pdf_to_images, encode_image_bytes,
    clean_text_for_preview, markdown_to_pdf
)

# DeepSeek OCR model name
DEEPSEEK_OCR_MODEL = "deepseek-ocr"

# DeepSeek OCR prompts based on input type
# Grounding prompts work for PDF documents but not for standalone images on Vertex AI
PROMPTS = {
    "document": "<|grounding|>Convert the document to markdown.",  # For PDFs (with image extraction)
    "image": "Free OCR.",  # For standalone images (grounding causes issues)
}


def parse_grounding_refs(text):
    """
    Parse DeepSeek-OCR grounding references from response text.
    
    Supports two formats:
    1. Self-hosted format: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>
    2. Vertex AI format: label[[x1, y1, x2, y2]]
    
    Returns list of (label, coordinates_list) tuples
    """
    refs = []
    
    # Try self-hosted format first: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
    pattern1 = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches1 = re.findall(pattern1, text, re.DOTALL)
    
    for label, coords_str in matches1:
        try:
            coords_list = eval(coords_str)
            # Ensure it's a list of lists
            if coords_list and not isinstance(coords_list[0], list):
                coords_list = [coords_list]
            refs.append((label.strip(), coords_list))
        except Exception as e:
            print(f"   ⚠️  Could not parse coordinates (format 1): {coords_str[:50]}... - {e}")
            continue
    
    # Try Vertex AI format: label[[x1, y1, x2, y2]]
    # Pattern matches: word followed by [[numbers]]
    pattern2 = r'\b(image|figure|fig|picture|photo|diagram|chart|graph|illustration|table)\s*\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
    matches2 = re.findall(pattern2, text, re.IGNORECASE)
    
    for match in matches2:
        label = match[0]
        coords = [int(match[1]), int(match[2]), int(match[3]), int(match[4])]
        refs.append((label.lower(), [coords]))
    
    return refs


def extract_images_from_refs(pil_image, refs, page_num=0):
    """
    Extract images from PIL image using grounding coordinates.
    
    DeepSeek-OCR uses normalized coordinates (0-999 range).
    
    Returns dict mapping image IDs to base64 encoded images.
    """
    images = {}
    image_width, image_height = pil_image.size
    
    img_idx = 0
    for label, coords_list in refs:
        if label.lower() in ('image', 'figure', 'fig', 'picture', 'photo', 'diagram', 'chart', 'graph', 'illustration'):
            for coords in coords_list:
                try:
                    x1, y1, x2, y2 = coords
                    
                    # Convert from 0-999 normalized coords to actual pixels
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    # Crop the image
                    cropped = pil_image.crop((x1, y1, x2, y2))
                    
                    # Convert to base64
                    img_buffer = BytesIO()
                    cropped.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    
                    # Create unique ID
                    img_id = f"page{page_num}_img{img_idx}"
                    images[img_id] = img_base64
                    img_idx += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Could not extract image at {coords}: {e}")
                    continue
    
    return images


def clean_grounding_tags(text, page_num=0):
    """Remove grounding tags from text and replace image refs with markdown"""
    img_idx = 0
    
    def replace_image_ref_format1(match):
        nonlocal img_idx
        label = match.group(1)
        if label.lower() in ('image', 'figure', 'fig', 'picture', 'photo', 'diagram', 'chart', 'graph', 'illustration'):
            img_id = f"page{page_num}_img{img_idx}"
            img_idx += 1
            return f"![{label}]({img_id})"
        return ""  # Remove non-image refs
    
    def replace_image_ref_format2(match):
        nonlocal img_idx
        label = match.group(1)
        img_id = f"page{page_num}_img{img_idx}"
        img_idx += 1
        return f"![{label}]({img_id})"
    
    # Replace self-hosted format: <|ref|>image<|/ref|><|det|>...<|/det|>
    cleaned = re.sub(
        r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>',
        replace_image_ref_format1,
        text,
        flags=re.DOTALL
    )
    
    # Replace Vertex AI format: image[[x1, y1, x2, y2]]
    cleaned = re.sub(
        r'\b(image|figure|fig|picture|photo|diagram|chart|graph|illustration)\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]',
        replace_image_ref_format2,
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Remove other grounding tags (text, title, sub_title, etc.) but keep the content
    # Pattern: label[[x1, y1, x2, y2]] followed by content
    cleaned = re.sub(
        r'\b(text|title|sub_title|header|footer|caption|paragraph)\s*\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]\s*',
        '',
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def ocr_single_image(client, base64_data, mime_type, prompt):
    """Run OCR on a single image and return the response"""
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


def extract_text_with_deepseek_ocr(file_path):
    """Extract text and images from image or PDF using DeepSeek OCR"""
    file_type = get_file_type(file_path)
    
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
    
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_images = {}
    
    if file_type == 'image':
        print(f"📸 Encoding image: {file_path}")
        base64_data = encode_file(file_path)
        mime_type = get_mime_type(file_path)
        print(f"✅ File encoded ({len(base64_data)} characters)")
        
        print(f"🚀 Sending to DeepSeek OCR model: {DEEPSEEK_OCR_MODEL}")
        print(f"   Prompt: {prompt}")
        
        response = ocr_single_image(client, base64_data, mime_type, prompt)
        extracted_text = response.choices[0].message.content
        
        total_usage["prompt_tokens"] = response.usage.prompt_tokens
        total_usage["completion_tokens"] = response.usage.completion_tokens
        total_usage["total_tokens"] = response.usage.total_tokens
        
    else:
        # PDF - convert to images first, use grounding for image extraction
        print(f"📄 Converting PDF to images: {file_path}")
        page_images = pdf_to_images(file_path)
        print(f"✅ PDF converted to {len(page_images)} page image(s)")
        
        print(f"🚀 Sending to DeepSeek OCR model: {DEEPSEEK_OCR_MODEL}")
        print(f"   Prompt: {prompt}")
        print(f"   Processing {len(page_images)} page(s)...")
        
        all_text = []
        
        for page_num, (img_bytes, mime_type) in enumerate(page_images, 1):
            print(f"\n   📄 Processing page {page_num}/{len(page_images)}...")
            
            base64_data = encode_image_bytes(img_bytes)
            response = ocr_single_image(client, base64_data, mime_type, prompt)
            raw_text = response.choices[0].message.content
            
            total_usage["prompt_tokens"] += response.usage.prompt_tokens
            total_usage["completion_tokens"] += response.usage.completion_tokens
            total_usage["total_tokens"] += response.usage.total_tokens
            
            # Parse grounding refs and extract images for this page
            refs = parse_grounding_refs(raw_text)
            if refs:
                print(f"      📍 Found {len(refs)} grounding references")
                # Load page image to extract cropped regions
                pil_image = PILImage.open(BytesIO(img_bytes))
                page_images_extracted = extract_images_from_refs(pil_image, refs, page_num=page_num)
                if page_images_extracted:
                    print(f"      🖼️  Extracted {len(page_images_extracted)} images")
                    all_images.update(page_images_extracted)
            
            # Clean the text for this page
            cleaned_text = clean_grounding_tags(raw_text, page_num=page_num)
            
            all_text.append(cleaned_text)
            print(f"      ✅ Page {page_num} complete ({len(cleaned_text)} characters)")
        
        extracted_text = "\n\n--- Page Break ---\n\n".join(all_text)
    
    return extracted_text, all_images, total_usage


def main():
    if len(sys.argv) < 2:
        print("Usage: python deepseek_ocr_to_pdf.py <filename> [output.pdf]")
        print("Example: python deepseek_ocr_to_pdf.py scan.jpg")
        print("         python deepseek_ocr_to_pdf.py document.pdf extracted_document.pdf")
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
        print(f"STEP 1: Extracting text and images with DeepSeek OCR ({file_type.upper()})")
        print("="*60)
        
        extracted_text, images, usage_info = extract_text_with_deepseek_ocr(input_path)
        
        if not extracted_text.strip():
            print("⚠️  Warning: No text was extracted from the file")
            sys.exit(1)
        
        print(f"\n✅ Extracted {len(extracted_text)} characters of text")
        print(f"✅ Found {len(images)} embedded images")
        
        # Show preview
        print("\n📝 Text preview:")
        preview = clean_text_for_preview(extracted_text)[:500]
        print("-" * 40)
        print(preview + ("..." if len(extracted_text) > 500 else ""))
        print("-" * 40)
        
        # Step 2: Convert to PDF
        print("\n" + "="*60)
        print("STEP 2: Generating PDF")
        print("="*60)
        
        markdown_to_pdf(extracted_text, output_path, images, model_name=DEEPSEEK_OCR_MODEL)
        
        print(f"✅ PDF saved to: {output_path}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"   Input file:   {input_path} ({file_type})")
        print(f"   Output PDF:   {output_path}")
        print(f"   Text length:  {len(extracted_text)} characters")
        print(f"   Images:       {len(images)}")
        if usage_info:
            print(f"   OCR usage:    {usage_info}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
