# Document Data Extractor - OCR

This project is a powerful, flexible tool designed to extract information from **any type of document** (Claim Forms, Receipts, Invoices, Business Cards, Contracts, etc.) and convert it into structured JSON data.

It utilizes a hybrid approach:

1. **Local OCR**: Uses **Tesseract** (for images) and **pypdf/pdf2image** (for PDFs) to extract raw text from documents locally on your machine.
2. **AI Parsing**: Sends the extracted text to **Google Gemini** (via `google-genai` SDK) to intelligently understand the context and structure the data into JSON.

## Features

- **Multi-Format Support**:
  - Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`)
  - PDFs (Both text-based digital PDFs and scanned image-based PDFs)
  - Word Documents (`.docx`)
- **Universal Extraction**: It detects the document type automatically and extracts relevant fields.
  - *Receipts*: Store name, date, total, line items.
  - *Claim Forms*: Patient name, diagnosis codes, dates, amounts.
  - *Invoices*: Vendor details, invoice #, total due.
- **Structured Output**: Returns clean, standard JSON.

## Prerequisites

Before running the script, you must install the following external tools on your system:

1. **Tesseract OCR** (Required for Images & Scanned PDFs)

   - Download (Windows): [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install it and **add the installation folder** (e.g., `C:\Program Files\Tesseract-OCR`) to your System **PATH**.

## Installation

1. **Clone the repository** (if applicable).
2. **Install Python Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Setup**:

   - Create a `.env` file in the root directory.
   - Add your Google Gemini API Key:
     ```env
     GEMINI_API_KEY=your_actual_api_key_here
     ```

## Usage

Run the script from the command line by providing the path to your document.

```bash
# For an Image
python extract_claim_data.py data/receipt.jpg

# For a PDF
python extract_claim_data.py data/invoice.pdf --output my_invoice.json
```

### Arguments

- `file_path` (Required): Path to the input file.
- `--output`, `-o`: (Optional) Custom path for the output JSON file. If not provided, it saves as `<filename>_extracted.json` in the same directory.
