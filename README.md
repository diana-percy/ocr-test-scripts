# OCR Test Scripts

Test scripts for validating OCR integration via LiteLLM.

## Prerequisites

```bash
pip install requests reportlab pillow python-dotenv
```

## Configuration

1. Copy the example environment file and add your credentials:

```bash
cp .env.example .env
```

2. Edit `.env` with your API key and base URL:

```bash
# .env
API_KEY=your-api-key
BASE_URL=https://api.com
```

3. Optionally update `OCR_MODEL` in `config.py`:

```python
# config.py
OCR_MODEL = "mistral-ocr-2505"  # or any other OCR model
```

### Supported OCR Models

Update `OCR_MODEL` in `config.py` to test different providers:

| Provider    | Model Name         |
| ----------- | ------------------ |
| Mistral     | `mistral-ocr-2505` |
| Deepseek AI | `deepseek-ocr`     |

## Scripts

Place your input files in the `resources/` folder. Generated PDFs will be saved to `output/`.

### 1. Test OCR with Images

Extract text from images (JPG, PNG, etc.):

```bash
python test_ocr_image.py <image_file>
```

### 2. Test OCR with PDFs

Extract text from PDF documents:

```bash
python test_ocr_pdf.py <pdf_file>
python test_ocr_pdf.py <pdf_file> --include-images  # Include base64 image embeddings in response
```

### 3. Convert Image/PDF to Searchable PDF

Extract text and images from an image or PDF and generate a formatted PDF:

```bash
python ocr_to_pdf.py <input_file> [output.pdf]

# Examples:
python ocr_to_pdf.py scan.jpg                    # Creates output/scan_ocr.pdf
python ocr_to_pdf.py document.pdf out.pdf        # Creates output/out.pdf
```

Supported input formats: jpg, jpeg, png, gif, webp, pdf

The output PDF includes:

- **Headers** properly formatted (H1, H2, H3)
- **Tables** with proper styling and alternating row colors
- **Images** embedded in their correct positions in the document
- **Lists** (bullet and numbered)
- **Inline formatting** (bold, italic, code)

## What to Expect

### Successful Response (test_ocr_image.py)

```
python test_ocr_image.py receipt.jpeg
📸 Encoding image: resources/receipt.jpeg
✅ Image encoded (141940 characters)

🚀 Sending to OCR model: mistral-ocr-2505

============================================================
RAW API RESPONSE
============================================================
{
  "pages": [
    {
      "index": 0,
      "markdown": "SHOPPING STORE\n03.22 PM\nREG 12-21\nCLERK 2\n1 MISC.\n1 STUFF\nSUBTOTAL\nTAX\nTOTAL\nCASH\nCHANGE\n00.48\n$7.99\n$8.48\n$0.74\nZZ\n$10.00\n$0.78\n\nNO REFUNDS\nNO EXCHANGES\nNO RETURNS",
      "images": [],
      "dimensions": {
        "dpi": 200,
        "height": 805,
        "width": 800
      }
    }
  ],
  "model": "mistral-ocr-2505",
  "document_annotation": null,
  "usage_info": {
    "pages_processed": 1,
    "doc_size_bytes": 106454
  },
  "object": "ocr"
}
```

### Successful Response (ocr_to_pdf.py)

```
============================================================
STEP 1: Extracting text and images with OCR (IMAGE)
============================================================
📸 Encoding image: resources/receipt.jpeg
✅ File encoded (141940 characters)
🚀 Sending to OCR model: mistral-ocr-2505
✅ Extracted 164 characters of text
✅ Found 0 embedded images

📝 Text preview:
----------------------------------------
SHOPPING STORE
03.22 PM
REG 12-21
CLERK 2
1 MISC.
1 STUFF
SUBTOTAL
TAX
TOTAL
----------------------------------------

============================================================
STEP 2: Generating PDF
============================================================
   Converting to PDF...
✅ PDF saved to: output/receipt_ocr.pdf

============================================================
SUMMARY
============================================================
   Input file:   resources/receipt.jpeg (image)
   Output PDF:   output/receipt_ocr.pdf
   Text length:  164 characters
   Images:       0
   OCR usage:    {'pages_processed': 1, 'doc_size_bytes': 106454}
```

## Error Handling

If you see errors, check:

- The OCR model is enabled/configured in LiteLLM
- LiteLLM version is v1.79.3 or later (OCR endpoint support)
- Your API key is valid
- The image/PDF file exists and is readable

## Response Format

Most OCR models return text in **Markdown format** with:

- Headers preserved (`# Title`, `## Heading`)
- Tables properly formatted
- Image placeholders with IDs
- Page-by-page extraction for multi-page documents

The scripts try multiple field names (`markdown`, `text`, `content`) to support different OCR providers.
