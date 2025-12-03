#!/usr/bin/env python3
"""
Convert images or PDFs to searchable/OCR-processed PDFs using OCR via LiteLLM

Usage: python ocr_to_pdf.py <filename> [output.pdf]
Example: python ocr_to_pdf.py scan.jpg
         python ocr_to_pdf.py document.pdf extracted_document.pdf

Input files are automatically read from the 'resources' folder.
Output files are automatically saved to the 'output' folder.

Supported input formats:
- Images: jpg, jpeg, png, gif, webp
- Documents: pdf

This script:
1. Sends the image/PDF to the configured OCR model for text extraction
2. Outputs the raw markdown text to a PDF with embedded images

Configure the OCR model and API settings in config.py

Required packages:
    pip install requests reportlab pillow
"""

import requests
import base64
import sys
import os
import re
from io import BytesIO

from utils import resolve_input_path, resolve_output_path, RESOURCES_DIR, OUTPUT_DIR

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from PIL import Image as PILImage
except ImportError:
    print("❌ Missing required libraries: reportlab and/or pillow")
    print("   Install them with: pip install reportlab pillow")
    sys.exit(1)

from config import API_KEY, BASE_URL, OCR_MODEL


def encode_file(file_path):
    """Encode file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_file_type(file_path):
    """Determine file type from extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    pdf_extensions = {'.pdf'}
    
    if ext in image_extensions:
        return 'image'
    elif ext in pdf_extensions:
        return 'pdf'
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {image_extensions | pdf_extensions}")


def get_mime_type(file_path):
    """Get MIME type for the file"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf'
    }
    return mime_types.get(ext, 'application/octet-stream')


def extract_text_with_ocr(file_path):
    """Extract text and images from image or PDF using OCR"""
    file_type = get_file_type(file_path)
    mime_type = get_mime_type(file_path)
    
    if file_type == 'image':
        print(f"📸 Encoding image: {file_path}")
    else:
        print(f"📄 Encoding PDF: {file_path}")
    
    base64_data = encode_file(file_path)
    print(f"✅ File encoded ({len(base64_data)} characters)")

    print(f"🚀 Sending to OCR model: {OCR_MODEL}")
    
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
            "model": OCR_MODEL,
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


def clean_text_for_preview(markdown_text):
    """Clean markdown text for preview"""
    # Remove markdown image references
    cleaned = re.sub(r'!\[.*?\]\(.*?\)\n?', '', markdown_text)
    # Remove multiple consecutive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def create_image_flowable(img_id, images, max_width=6*inch, max_height=6*inch):
    """Create a reportlab Image from base64 data"""
    if img_id not in images:
        return None
    
    try:
        img_base64 = images[img_id]
        
        # Handle data URL format
        if img_base64.startswith('data:'):
            img_base64 = img_base64.split(',', 1)[1]
        
        # Decode image
        img_data = base64.b64decode(img_base64)
        img_buffer = BytesIO(img_data)
        
        # Get dimensions using PIL
        pil_img = PILImage.open(BytesIO(img_data))
        orig_width, orig_height = pil_img.size
        
        # Calculate scaled dimensions
        width_ratio = max_width / orig_width
        height_ratio = max_height / orig_height
        scale = min(width_ratio, height_ratio, 1.0)
        
        display_width = orig_width * scale
        display_height = orig_height * scale
        
        img_buffer.seek(0)
        return RLImage(img_buffer, width=display_width, height=display_height)
        
    except Exception as e:
        print(f"⚠️  Could not create image {img_id}: {e}")
        return None


def markdown_to_pdf(markdown_text, output_path, images=None):
    """Convert markdown text with images to PDF - simple text output with embedded images"""
    
    images = images or {}
    styles = getSampleStyleSheet()
    
    # Simple text style
    text_style = ParagraphStyle(
        name='Text',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )
    
    footer_style = ParagraphStyle(
        name='Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        alignment=TA_CENTER
    )
    
    # Create the document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    flowables = []
    
    # Split by image references and process
    # Pattern to find image references like ![alt](img_id)
    pattern = r'(!\[.*?\]\(.*?\))'
    parts = re.split(pattern, markdown_text)
    
    print("   Converting to PDF...")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Check if this is an image reference
        img_match = re.match(r'!\[.*?\]\((.+?)\)', part)
        if img_match:
            img_id = img_match.group(1)
            img_flowable = create_image_flowable(img_id, images)
            if img_flowable:
                flowables.append(Spacer(1, 6))
                flowables.append(img_flowable)
                flowables.append(Spacer(1, 6))
        else:
            # Regular text - escape HTML special chars and add as paragraph
            text = part.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            # Replace newlines with <br/> for proper line breaks
            text = text.replace('\n', '<br/>')
            if text:
                flowables.append(Paragraph(text, text_style))
    
    # Add footer
    flowables.append(Spacer(1, 24))
    flowables.append(Paragraph(
        f"Generated from OCR extraction using {OCR_MODEL} via LiteLLM",
        footer_style
    ))
    
    # Build the PDF
    doc.build(flowables)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_to_pdf.py <filename> [output.pdf]")
        print("Example: python ocr_to_pdf.py scan.jpg")
        print("         python ocr_to_pdf.py document.pdf extracted_document.pdf")
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
        print(f"STEP 1: Extracting text and images with OCR ({file_type.upper()})")
        print("="*60)
        
        markdown_text, images, usage_info = extract_text_with_ocr(input_path)
        
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
        
        markdown_to_pdf(markdown_text, output_path, images)
        
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
