import os
import argparse
import logging
import json
from typing import List, Union
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import fitz
import docx
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_TESSERACT_PATH = r'path to tessaract.exe'
if os.path.exists(DEFAULT_TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_PATH

class DataExtractionPipeline:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logging.warning("No API Key found. Intelligent parsing step might fail.")
        
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logging.error(f"Failed to initialize Gemini client: {e}")

    def run_pipeline(self, input_path: str, output_path: str = None):
        """
        Executes the extraction pipeline:
        Input -> Render -> Preprocess -> OCR -> Parse -> Output
        """
        logging.info(f"Starting pipeline for: {input_path}")
        
        pages = self.render_pages(input_path)
        extracted_text_blocks = []
        
        for i, page_image in enumerate(pages):
            logging.info(f"Processing page {i+1}/{len(pages)}...")
            processed_image = self.preprocess_image(page_image)
            text = self.extract_text_ocr(processed_image)
            extracted_text_blocks.append(f"--- Page {i+1} ---\n{text}")

        full_text = "\n\n".join(extracted_text_blocks)
        
        markdown_content = self.intelligent_parse(full_text)
        
        self.save_output(markdown_content, input_path, output_path)
        return markdown_content

    def render_pages(self, file_path: str) -> List[Image.Image]:
        ext = os.path.splitext(file_path)[1].lower()
        images = []
        try:
            if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
                images.append(Image.open(file_path).convert('RGB'))
            elif ext == '.pdf':
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                doc.close()
            else:
                logging.warning(f"Unsupported format for rendering: {ext}. Attempting text extraction fallback.")
        except Exception as e:
            logging.error(f"Error rendering pages: {e}")
        return images

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Pre-process image for better OCR results:
        - Grayscale
        - Contrast enhancement
        - Thresholding (Binarization)
        """

        gray_img = ImageOps.grayscale(image)
        enhancer = ImageEnhance.Contrast(gray_img)
        contrast_img = enhancer.enhance(2.0)

        threshold_img = contrast_img.point(lambda p: 255 if p > 128 else 0)
        return threshold_img

    def extract_text_ocr(self, image: Image.Image) -> str:
        """
        Uses Tesseract OCR to extract text from the preprocessed image.
        Uses layout preservation config.
        """

        custom_config = r'--oem 3 --psm 3'
        return pytesseract.image_to_string(image, config=custom_config)

    def _extract_text_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                full_text.append(" | ".join(row_text))
        return "\n".join(full_text)

    def intelligent_parse(self, raw_text: str) -> str:
        if not self.client:
            return raw_text

        # Prompt to preserve content and structure
        prompt = f"""
        You are a strict Document Formatter AI.
        
        Your Goal:
        Format the provided raw OCR text into clean **Markdown**, strictly preserving the original content and structure.
        
        Strict Rules:
        1. **NO Content Modification**: Do NOT rewrite, summarize, or change the wording. Fix ONLY obvious OCR character errors (e.g., '1l' -> 'll' or 'rn' -> 'm') but do not alter the meaning or words.
        2. **Tabular Data**: Detect table structures (rows and columns) in the text and format them as proper Markdown tables.
           - Ensure columns are correctly aligned.
           - If a section looks like a table or aligned key-value pairs, treat it as a table.
        3. **Aligned Data**: If data is visually aligned (like "Name:     John Doe"), preserve this relationship, preferably using a table or a list.
        4. **Structure**: Maintain headers, lists, and sections exactly as they appear.
        5. **Markings**: If you see text indicating checkboxes (e.g., "[ ]", "[x]", "Yes/No" selection), preserve the state as Markdown `- [ ]` or `- [x]`.
        6. **Output**: Return ONLY the Markdown string. No conversational filler.

        Raw Text:
        -----------------------
        {raw_text}
        -----------------------
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            logging.error(f"Intelligent Parsing failed: {e}")
            return raw_text

    def save_output(self, content: str, input_path: str, output_path: str = None):
        if not output_path:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_test_processed.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.info(f"Successfully saved output to: {output_path}")

    def run_pipeline_smart(self, input_path: str, output_path: str = None):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.docx', '.doc']:
            logging.info("Processing as DOCX (Direct Text Extraction)")
            raw_text = self._extract_text_docx(input_path)
            markdown_content = self.intelligent_parse(raw_text)
            self.save_output(markdown_content, input_path, output_path)
            return markdown_content
        else:
            return self.run_pipeline(input_path, output_path)

def main():
    parser = argparse.ArgumentParser(description="Test Pipeline for Strict Formatting")
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("--output", "-o", help="Path to output markdown file")
    
    args = parser.parse_args()
    
    pipeline = DataExtractionPipeline()
    pipeline.run_pipeline_smart(args.input_file, args.output)

if __name__ == "__main__":
    main()