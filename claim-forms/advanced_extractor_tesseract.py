import os
import cv2
import json
import logging
import numpy as np
import pandas as pd
import pytesseract
import imutils
import torch
import docx
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================================
# CONFIGURATION
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
load_dotenv()

USE_GPU = torch.cuda.is_available()

# ==========================================================
# FILE CONVERSION & EXTRACTION
# ==========================================================

def pdf_to_images(pdf_path, dpi=300):
    logging.info("Converting PDF to images: %s", pdf_path)
    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        images = []
        for page in pages:
            # Convert PIL image to OpenCV BGR format
            images.append(cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR))
        return images
    except Exception as e:
        logging.error("Failed to convert PDF %s: %s", pdf_path, e)
        return []

def extract_text_from_docx(docx_path):
    logging.info("Extracting text from DOCX: %s", docx_path)
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        logging.error("Failed to extract DOCX %s: %s", docx_path, e)
        return ""

# ==========================================================
# IMAGE PROCESSING & OCR
# ==========================================================

def correct_orientation(image):
    try:
        osd = pytesseract.image_to_osd(image)
        angle = int(osd.split("Rotate: ")[1].split("\n")[0])
        if angle != 0:
            image = imutils.rotate_bound(image, angle)
            logging.info("Coarse rotation corrected: %d degrees", angle)
    except Exception:
        pass
    return image

def deskew(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        if abs(angle) > 0.5:
            image = imutils.rotate_bound(image, angle)
            logging.info("Fine skew corrected: %.2f degrees", angle)
    except Exception as e:
        logging.warning("Deskewing failed: %s", e)
    return image

def process_single_image(image):
    """Processes a single image (page) using Tesseract OCR."""
    # Optional pre-processing
    image = correct_orientation(image)
    image = deskew(image)
    
    logging.info("Running Tesseract OCR detection and recognition...")
    page_text = pytesseract.image_to_string(image)
            
    return page_text.strip()

# ==========================================================
# GEMINI STRUCTURING
# ==========================================================

def structure_with_gemini(text, filename):
    logging.info("Sending extracted text to Gemini for structuring: %s", filename)
    client = genai.Client()

    system_prompt = """
    You are a professional medical data extractor. Your task is to extract all the fields from the provided medical claim form text.
    
    Extract every single field and its value present in the input file text (strictly same values as of input file, no hallucinated or guessed values).

    Instructions:
    - If a field is not found, leave it out.
    - If multiple values exist for a field, return them as a list.
    - Return the data in a valid JSON format as a flat or nested dictionary.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json"
            ),
            contents=f"Filename: {filename}\nInput Text:\n{text}"
        )
        return json.loads(response.text.strip())
    except Exception as e:
        logging.error("Gemini structuring failed: %s", e)
        return {}

# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main_pipeline(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    full_text = ""
    
    if ext == ".pdf":
        images = pdf_to_images(file_path)
        if images:
            logging.info("Processing %d pages sequentially...", len(images))
            results = []
            for i, img in enumerate(images):
                logging.info("Processing page %d/%d", i+1, len(images))
                results.append(process_single_image(img))
            full_text = "\n".join(results)
    elif ext == ".docx":
        full_text = extract_text_from_docx(file_path)
    elif ext in [".tif", ".tiff"]:
        logging.info("Processing TIFF image (potentially multipage): %s", file_path)
        try:
            img = Image.open(file_path)
            results = []
            for i in range(getattr(img, 'n_frames', 1)):
                img.seek(i)
                frame = np.array(img.convert('RGB'))
                cv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                logging.info("Processing TIFF page %d", i+1)
                results.append(process_single_image(cv_img))
            full_text = "\n".join(results)
        except Exception as e:
            logging.error("Failed to process TIFF %s: %s", file_path, e)
    elif ext in [".png", ".jpg", ".jpeg"]:
        logging.info("Processing single image: %s", file_path)
        image = cv2.imread(file_path)
        if image is not None:
            full_text = process_single_image(image)
        else:
            logging.error("Could not read image file: %s", file_path)
    else:
        logging.warning("Unsupported file format: %s", ext)
        return

    if not full_text:
        logging.error("No text could be extracted from %s", file_path)
        return

    # Structure data
    structured_data = structure_with_gemini(full_text, os.path.basename(file_path))
    
    # Save JSON
    output_base = os.path.splitext(os.path.basename(file_path))[0]
    json_path = f"output/{output_base}_structured.json"
    os.makedirs("output", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(structured_data, f, indent=4)
    logging.info("Saved JSON output to %s", json_path)

    # Save CSV
    if structured_data:
        csv_path = f"output/{output_base}_structured.csv"
        # Flatten lists for CSV if necessary or just store as string
        data_for_df = structured_data.copy()
        for k, v in data_for_df.items():
            if isinstance(v, (list, dict)):
                data_for_df[k] = json.dumps(v)
        
        df = pd.DataFrame([data_for_df])
        df.to_csv(csv_path, index=False)
        logging.info("Saved CSV output to %s", csv_path)
        
        print("\n--- Extracted Data ---")
        print(df.to_string(index=False))

import argparse

def main():
    parser = argparse.ArgumentParser(description="Universal Medical Claim Form Extractor using Tesseract and Gemini")
    parser.add_argument("input", help="Path to an input file (PDF, DOCX, Image) or a directory containing files")
    
    args = parser.parse_args()
    input_path = args.input

    if not os.path.exists(input_path):
        logging.error("The provided path does not exist: %s", input_path)
        return

    if os.path.isdir(input_path):
        logging.info("Scanning directory: %s", input_path)
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        if not files:
            logging.warning("No files found in directory: %s", input_path)
            return
        
        for filename in files:
            file_full_path = os.path.join(input_path, filename)
            main_pipeline(file_full_path)
    else:
        # Single file
        main_pipeline(input_path)

if __name__ == "__main__":
    main()
