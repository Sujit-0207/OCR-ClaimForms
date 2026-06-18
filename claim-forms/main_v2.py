import os
import argparse
import json
import logging
from dotenv import load_dotenv
from google import genai
from PIL import Image
import pytesseract
import pypdf
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables
load_dotenv()

# NOTE: You must have Tesseract-OCR installed on your system and in your PATH.
# If not in PATH, uncomment and set the path below:
# pytesseract.pytesseract.tesseract_cmd = r'path to tesseract.exe'

def get_text_from_image(image_path: str) -> str:
    """Extracts text from an image using Pytesseract."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        logging.error(f"OCR failed for {image_path}. Ensure Tesseract is installed. Error: {e}")
        return ""

def get_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF. 
    First tries standard text extraction. 
    If that yields little text (scanned PDF), converts pages to images and uses OCR.
    """
    text_content = []
    
    # Method 1: standard text extraction (pypdf)
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text_content.append(page.extract_text() or "")
    except Exception as e:
        logging.warning(f"Standard PDF text extraction failed: {e}")
    
    raw_text = "\n".join(text_content)
    
    # Check if we got meaningful text. If less than 50 chars, assume it's scanned/image-based
    if len(raw_text.strip()) < 50:
        print("Note: PDF seems to be scanned (little text found). Switching to OCR...")
        try:
            # Method 2: OCR (pdf2image + tesseract)
            images = convert_from_path(pdf_path)
            ocr_text = []
            for i, img in enumerate(images):
                print(f"OCR processing page {i+1}...")
                ocr_text.append(pytesseract.image_to_string(img))
            
            return "\n".join(ocr_text)
            
        except ImportError:
            logging.error("pdf2image or poppler not found. For scanned PDFs, please install 'poppler-utils'.")
        except Exception as e:
            logging.error(f"OCR for PDF failed: {e}. (Do you have Poppler installed and in PATH?)")
            
            if raw_text:
                return raw_text
            return ""
            
    return raw_text

def process_content_with_gemini(extracted_text: str, file_type: str, file_path: str) -> str:
    """
    Sends extracted text to Gemini to be structured into Markdown.
    """
    if not extracted_text.strip():
        return "Error: No text could be extracted from the file."

    prompt = f"""
    You are an intelligent data extraction assistant.
    I will provide you with unstructured text extracted from a {file_type}.
    
    Your task:
    1. Analyze the text to understand what kind of document it is (Claim Form, Receipt, Invoice, Lab Report, etc.).
    2. Extract key information relevant to that document type instructed below.
    3. Structure the output into a valid mark-down object.

    The mark-down file must include data in the below format (strictly tabular format):
    Filename (strictly include only the file name without path) {file_path} | form type (HCFA/ UB-04)| Accept assignment (Yes if 'Y' or tick symbol is present/ No if 'N' or empty or x is present) (Box 27 in HCFA and Box 53 in UB-04) | Amount Paid (check amount in box 29 and extract it correct in HCFA) or Total Charges (check for correct total sum from Box 47 in UB-04)

    Here is the text:
    --------------------------------------------------
    {extracted_text}
    --------------------------------------------------

    Return ONLY the raw mark-down object as file. 
    """

    try:
        client = genai.Client()
                
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Clean up response text
        text_content = response.text.strip()
        if text_content.startswith("```"):
            # Strip opening code block
            first_newline = text_content.find("\n")
            if first_newline != -1:
                text_content = text_content[first_newline+1:]
            else:
                 # Fallback if just ``` without newline
                text_content = text_content[3:]
        
        if text_content.endswith("```"):
            text_content = text_content[:-3]
            
        return text_content.strip()
        
    except Exception as e:
        logging.error(f"Gemini processing failed: {e}")
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Extract content from files (Img, PDF, Docx) and structure with Gemini.")
    parser.add_argument("file_path", help="Path to the input file (Image, PDF, DOCX)")
    parser.add_argument("--output", "-o", help="Path to save the output file")
    
    args = parser.parse_args()
    file_path = args.file_path
    
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    # Determin file type and extract text
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    file_type_label = "Document"

    print(f"Processing file: {file_path}")

    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        file_type_label = "Image"
        print("Extracting text from Image using Tesseract...")
        extracted_text = get_text_from_image(file_path)
    elif ext == '.pdf':
        file_type_label = "PDF"
        print("Extracting text from PDF...")
        extracted_text = get_text_from_pdf(file_path)
    else:
        logging.error(f"Unsupported file type: {ext}")
        return

    if not extracted_text:
        logging.warning("No text extracted. The file might be empty or scanned without OCR capabilities (for PDFs).")
    else:
        print(f"Extracted {len(extracted_text)} characters. Sending to Gemini...")

    # Structure with Gemini
    structured_data = process_content_with_gemini(extracted_text, file_type_label, file_path)
    
    # Output results
    print("\n--- Structured Data (Markdown) ---")
    print(structured_data)
    
    # Save to file
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = f"{base_name}_extracted.md"
        
    with open(output_file, 'w') as f:
        f.write(structured_data)
        print(f"\nSaved to: {output_file}")

if __name__ == "__main__":
    main()
