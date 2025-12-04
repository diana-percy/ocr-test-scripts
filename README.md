# OCR Test Scripts

Test scripts for validating OCR integration via LiteLLM.

## Prerequisites

```bash
pip install requests reportlab pillow python-dotenv openai pymupdf
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

## Supported OCR Models

This repo contains model-specific test scripts for different OCR providers:

| Provider    | Model Name         | Endpoint               | Script(s)                                        |
| ----------- | ------------------ | ---------------------- | ------------------------------------------------ |
| Mistral     | `mistral-ocr-2505` | `/v1/ocr`              | `test_mistral_ocr.py`, `mistral_ocr_to_pdf.py`   |
| DeepSeek AI | `deepseek-ocr`     | `/v1/chat/completions` | `test_deepseek_ocr.py`, `deepseek_ocr_to_pdf.py` |

**Note:** Mistral OCR uses the dedicated `/v1/ocr` endpoint, while DeepSeek-OCR uses the standard `/v1/chat/completions` endpoint with multimodal messages.

## Scripts

Place your input files in the `resources/` folder. Generated PDFs will be saved to `output/`.

### Mistral OCR Scripts

These scripts use the `/v1/ocr` endpoint with Mistral's OCR model.

#### Test OCR (Images and PDFs)

```bash
python test_mistral_ocr.py <file>
python test_mistral_ocr.py <file> --include-images  # Include base64 image embeddings
```

#### Convert to Searchable PDF

```bash
python mistral_ocr_to_pdf.py <input_file> [output.pdf]

# Examples:
python mistral_ocr_to_pdf.py scan.jpg                    # Creates output/scan_ocr.pdf
python mistral_ocr_to_pdf.py document.pdf out.pdf        # Creates output/out.pdf
```

### DeepSeek OCR Scripts

These scripts use the `/v1/chat/completions` endpoint with DeepSeek's OCR model.

**Notes:**

- For PDF input, pages are automatically converted to images first using PyMuPDF, then OCR'd page by page.
- **Test script** (`test_deepseek_ocr.py`): Uses `Free OCR.` prompt for simple text extraction.
- **PDF conversion script** (`deepseek_ocr_to_pdf.py`): Uses grounding prompts for PDFs to extract embedded images:
  - PDFs: `<|grounding|>Convert the document to markdown.` (extracts images)
  - Images: `Free OCR.` (text only, grounding causes issues with standalone images)

#### Test OCR (Images and PDFs)

```bash
python test_deepseek_ocr.py <file>
```

#### Convert to Searchable PDF

```bash
python deepseek_ocr_to_pdf.py <input_file> [output.pdf]
```

## What to Expect

### Successful Response (test_mistral_ocr.py)

```
python test_mistral_ocr.py receipt.jpeg
📸 Encoding image: resources/receipt.jpeg
✅ File encoded (141940 characters)

🚀 Sending to Mistral OCR model: mistral-ocr-2505
   Using /v1/ocr endpoint

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
  "usage_info": {
    "pages_processed": 1,
    "doc_size_bytes": 106454
  },
  "object": "ocr"
}
```

### Successful Response (test_deepseek_ocr.py)

```
python test_deepseek_ocr.py receipt.jpeg
📸 Encoding image: resources/receipt.jpeg
✅ Image encoded (141940 characters)
   MIME type: image/jpeg

🚀 Sending to DeepSeek-OCR model: deepseek-ocr
   Using /v1/chat/completions endpoint
   Prompt: Free OCR.

============================================================
EXTRACTED TEXT
============================================================
12-21
REG
CLERK 2
1 MISC.
1 STUFF
SUBTOTAL
TAX
TOTAL
CASH
CHANGE
NO REFUNDS
NO EXCHANGES
NO RETURNS

03:22 PM
618

$0.49
$7.99
$8.48
$0.22
$10.00
$0.78
```

### Successful Response (mistral_ocr_to_pdf.py)

```
============================================================
STEP 1: Extracting text and images with Mistral OCR (IMAGE)
============================================================
📸 Encoding image: resources/receipt.jpeg
✅ File encoded (141940 characters)
🚀 Sending to Mistral OCR model: mistral-ocr-2505
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
- LiteLLM version is v1.79.3 or later (OCR endpoint support for Mistral)
- Your API key is valid
- The image/PDF file exists and is readable

## Response Format

### Mistral OCR (`/v1/ocr` endpoint)

Returns structured response with:

- `pages` array with `markdown` text per page
- `images` array with embedded images (when `--include-images` flag is used)
- `dimensions` with page size info
- `usage_info` with processing stats

### DeepSeek OCR (`/v1/chat/completions` endpoint)

Returns standard chat completion response with extracted text in `choices[0].message.content`.

**Known Issues with Vertex AI hosted DeepSeek-OCR:**

- Grounding prompts work for PDFs but cause issues with standalone images
- Some pages (especially blank or low-contrast pages) may return repeated `0.0.0.0...` patterns
- Use Mistral OCR for more reliable results with complex documents
