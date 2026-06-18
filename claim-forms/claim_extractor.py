import cv2
import numpy as np
import pytesseract
import json
import os
import logging
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load environment variables
load_dotenv()

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = 'Tesseract path'


def apply_deconvolution(image):
    """
    Applies a deconvolution-like sharpening filter to deblur the image.
    Using a Laplacian-based sharpening kernel which acts as a simple high-pass filter
    to restore edges in blurry images.
    """
    # 5x5 Deconvolution kernel (Sharpening)
    kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  2,  2,  2, -1],
        [-1,  2,  8,  2, -1],
        [-1,  2,  2,  2, -1],
        [-1, -1, -1, -1, -1]
    ]) / 8.0

    # Apply filter
    deblurred = cv2.filter2D(image, -1, kernel)

    # Blend with original to avoid over-sharpening noise
    return cv2.addWeighted(image, 1.5, deblurred, -0.5, 0)


def advanced_opencv_extraction(image_path):
    """
    Uses multiple advanced OpenCV techniques including Deconvolution/Sharpening
    to extract text with Pytesseract.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Deconvolution (Deblurring/Sharpening)
    deblurred = apply_deconvolution(gray)

    # 3. Scale up for better OCR
    resized = cv2.resize(deblurred, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 4. De-noise (Bilateral filter preserves edges)
    denoised = cv2.bilateralFilter(resized, 9, 75, 75)

    # 5. Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 6. Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Extract text from processed image
    text = pytesseract.image_to_string(processed)

    # Second OCR pass using Otsu threshold
    _, thresh_otsu = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    text_otsu = pytesseract.image_to_string(thresh_otsu)

    # Choose better result
    final_text = text if len(text) > len(text_otsu) else text_otsu
    return final_text


def process_with_gemini(extracted_text, file_path):
    """
    Sends extracted text to Gemini using a System Prompt to structure into JSON
    containing a markdown-formatted table.
    """
    if not extracted_text.strip():
        return {"error": "No text extracted"}

    filename = os.path.basename(file_path)

    system_prompt = f"""
    You are an intelligent data extraction assistant.
    I will provide you with unstructured text extracted from a medical document.

    Your task:
    1. Identify document type (Claim Form, Receipt, Invoice, Lab Report, etc.)
    2. Extract key details
    3. Return structured JSON.

    The JSON must include a "markdown_table" field with format:

    Filename | form type (HCFA/UB-04) | Accept assignment |
    Amount Paid (HCFA Box 29) OR Total Charges (UB-04 Box 47)

    CRITICAL:
    For UB-04, verify total charges by summing line items.
    Example: 235.00 + 72.00 = 307.00

    Return ONLY raw JSON with:
    - "markdown_table"
    - "extracted_fields"
    """

    user_content = f"""
    Filename: {filename}
    Extracted Text:
    ------------------------
    {extracted_text}
    ------------------------
    """

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json"
            ),
            contents=user_content
        )

        json_text = response.text.strip()
        return json.loads(json_text)

    except Exception as e:
        logging.error(f"Gemini processing failed: {e}")
        return {"error": str(e)}


def main():
    input_path = 'test.jpg'
    structured_output_path = 'test_structured_data.json'

    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Step 1: Extracting text using OpenCV from {input_path}...")

    try:
        text_content = advanced_opencv_extraction(input_path)

        logging.info("Step 2: Sending text to Gemini...")
        structured_data = process_with_gemini(text_content, input_path)

        if "error" in structured_data:
            print(f"Error: {structured_data['error']}")
            return

        # Save JSON output
        with open(structured_output_path, 'w') as f:
            json.dump(structured_data, f, indent=4)

        # Convert to DataFrame
        if "extracted_fields" in structured_data:
            fields = structured_data["extracted_fields"]
            df = pd.DataFrame([fields])

            print("\n--- Structured Data ---")
            print(df.to_string(index=False))

    except Exception as e:
        logging.error(f"Failed to process image: {e}")


if __name__ == "__main__":
    main()
