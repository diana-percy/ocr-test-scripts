"""
Shared utilities for OCR test scripts
"""

import os
import base64
import re
from io import BytesIO

# Directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(SCRIPT_DIR, "resources")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Supported file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
PDF_EXTENSIONS = {'.pdf'}


def pdf_to_images(pdf_path, dpi=144):
    """
    Convert PDF pages to images using PyMuPDF (fitz).
    Returns a list of (image_bytes, mime_type) tuples for each page.
    
    Requires: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
    
    images = []
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        
        # Convert to PNG bytes
        img_bytes = pixmap.tobytes("png")
        images.append((img_bytes, "image/png"))
    
    pdf_document.close()
    return images


def encode_image_bytes(image_bytes):
    """Encode image bytes to base64"""
    return base64.b64encode(image_bytes).decode('utf-8')


def get_file_type(file_path):
    """Determine file type from extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    elif ext in PDF_EXTENSIONS:
        return 'pdf'
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {IMAGE_EXTENSIONS | PDF_EXTENSIONS}")


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


def encode_file(file_path):
    """Encode file to base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def resolve_input_path(filename):
    """Resolve input file path - check resources folder if not found directly"""
    # If it's an absolute path or exists as-is, use it
    if os.path.isabs(filename) or os.path.exists(filename):
        return filename
    
    # Check in resources folder
    resources_path = os.path.join(RESOURCES_DIR, filename)
    if os.path.exists(resources_path):
        return resources_path
    
    # Return original (will fail with appropriate error later)
    return filename


def resolve_output_path(filename):
    """Resolve output file path - use output folder"""
    # If it's an absolute path, use it as-is
    if os.path.isabs(filename):
        return filename
    
    # Otherwise, put it in the output folder
    return os.path.join(OUTPUT_DIR, filename)


def clean_text_for_preview(markdown_text):
    """Clean markdown text for preview"""
    # Remove markdown image references
    cleaned = re.sub(r'!\[.*?\]\(.*?\)\n?', '', markdown_text)
    # Remove multiple consecutive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


# PDF Generation utilities - requires reportlab and pillow
def create_image_flowable(img_id, images, max_width, max_height):
    """Create a reportlab Image from base64 data
    
    Requires: reportlab, pillow
    """
    from reportlab.platypus import Image as RLImage
    from PIL import Image as PILImage
    
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


def markdown_to_pdf(markdown_text, output_path, images=None, model_name="OCR"):
    """Convert markdown text with images to PDF
    
    Requires: reportlab, pillow
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    
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
            img_flowable = create_image_flowable(img_id, images, max_width=6*inch, max_height=6*inch)
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
        f"Generated from OCR extraction using {model_name} via LiteLLM",
        footer_style
    ))
    
    # Build the PDF
    doc.build(flowables)

