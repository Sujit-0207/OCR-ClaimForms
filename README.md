# HCFA 1500 OCR & LLM Extractor

This Jupyter Notebook provides a complete pipeline to extract structured data from HCFA 1500 (CMS 1500) medical claim forms using Tesseract OCR and OpenAI's LLMs.

##  Features
- **OCR Engine**: Utilizes Tesseract for high-quality text and layout extraction.
- **AI Extraction**: Uses GPT-4o to parse messy OCR output into structured JSON.
- **Preprocessing**: Includes image enhancement steps (grayscale, contrast, sharpening) to improve accuracy.

##  Prerequisites
1. **OpenAI API Key**: Required for the LLM extraction phase.
2. **Form Image**: A scan or photo of an HCFA 1500 form (JPG/PNG).

##  How to Use
1. **Open in Colab**: Go to [Google Colab](https://colab.research.google.com) and upload the `hcfa-1500-ocr-llm-extractor.ipynb` file.
2. **Setup Dependencies**: Run the first cell to install Tesseract and required Python libraries.
3. **Configure**:
   - Paste your OpenAI API key into the configuration cell.
4. **Upload Form**:
   - Upload your HCFA 1500 image to the Colab files section (left sidebar).
   - Ensure the filename matches the path in the "Run on an Image" cell.
5. **Execute**: Run all cells to see the extracted JSON data.

##  Extracted Fields
The pipeline extracts key fields including:
- Patient & Insured Information (Boxes 1-5)
- Diagnosis Codes (Box 21)
- Service Lines (Box 24)
- Tax ID & Total Charges (Boxes 25, 28)
