# ============================================================
# DOCUMENT OCR + LLM EXTRACTION PIPELINE
# ============================================================
#
# OVERVIEW
# --------
# This pipeline performs the following operations:
#
# Step 1  : Convert PDF pages into images
# Step 2  : Preprocess images for better OCR accuracy
# Step 3  : Extract text using Tesseract OCR
# Step 4  : Save raw OCR text for debugging/auditing
# Step 5  : Use Gemini to extract structured information
# Step 6  : Validate and repair missing critical fields
# Step 7  : Convert structured output into tabular datasets
# Step 8  : Export results for downstream analysis
#
# Current Architecture:
#
# PDF
#  ↓
# Image Rendering
#  ↓
# Image Preprocessing
#  ↓
# OCR (Tesseract)
#  ↓
# Raw Text
#  ↓
# Gemini LLM Extraction
#  ↓
# Post Processing / Validation
#  ↓
# Structured Output
#  ↓
# CSV / Excel / Database
#
# ============================================================

# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import argparse
import logging
import json
import re

# Image processing libraries
from PIL import Image, ImageOps, ImageEnhance

# OCR Engine
import pytesseract

# PDF rendering
import fitz  # PyMuPDF

# DOCX support (future use)
import docx

# Environment variables
from dotenv import load_dotenv

# Gemini Client
from google import genai

# Data Processing
import pandas as pd

# Load environment variables
load_dotenv()

# ============================================================
# STEP 2: PDF PAGE RENDERING
# ============================================================
#
# PURPOSE:
# Convert each page of a PDF document into a high-resolution
# image suitable for OCR processing.
#
# WHY?
# OCR engines operate on images, not PDF pages.
#
# FUTURE IMPROVEMENT:
# - Multi-threaded rendering
# - TIFF support
# - Image quality checks
# ============================================================

def render_pages(file_path):

    images = []

    try:
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Render page at 2x resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            images.append(img)

        doc.close()

    except Exception as e:
        logging.error(f"PDF Rendering Error: {e}")

    return images


# ============================================================
# STEP 3: IMAGE PREPROCESSING
# ============================================================
#
# PURPOSE:
# Improve OCR accuracy before text extraction.
#
# PROCESS:
# 1. Convert image to grayscale
# 2. Increase contrast
# 3. Apply binary thresholding
#
# BENEFIT:
# Removes noise and improves text visibility.
#
# FUTURE IMPROVEMENTS:
# - CLAHE
# - Deskewing
# - Blur Detection
# - Adaptive Thresholding
# - Dynamic preprocessing selection
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    gray_img = ImageOps.grayscale(image)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(gray_img)
    contrast_img = enhancer.enhance(2.0)

    # Binary thresholding
    threshold_img = contrast_img.point(
        lambda p: 255 if p > 128 else 0
    )

    return threshold_img


# ============================================================
# STEP 4: OCR TEXT EXTRACTION
# ============================================================
#
# PURPOSE:
# Extract machine-readable text from images.
#
# CURRENT ENGINE:
# Tesseract OCR
#
# FUTURE IMPROVEMENT:
# Add fallback OCR strategy:
#
# Primary OCR  -> PaddleOCR
# Fallback OCR -> Tesseract
#
# OR
#
# If OCR confidence < threshold:
#     Reprocess image
#     Retry OCR
#
# BENEFIT:
# Higher reliability on low-quality scans.
# ============================================================

def extract_text_ocr(image):

    custom_config = r'--oem 3 --psm 3'

    return pytesseract.image_to_string(
        image,
        config=custom_config
    )


# ============================================================
# STEP 5: DOCUMENT OCR PIPELINE
# ============================================================
#
# FLOW:
#
# PDF
# ↓
# Render Pages
# ↓
# Preprocess Images
# ↓
# OCR Extraction
# ↓
# Raw Text Generation
#
# PURPOSE:
# Create complete text representation of document.
# ============================================================

input_file = "sample.pdf"

pages = render_pages(input_file)

raw_text_blocks = []

for i, page in enumerate(pages):

    processed_page = preprocess_image(page)

    text = extract_text_ocr(processed_page)

    raw_text_blocks.append(
        f"--- Page {i+1} ---\n{text}"
    )

raw_text = "\n\n".join(raw_text_blocks)


# ============================================================
# STEP 6: SAVE RAW OCR OUTPUT
# ============================================================
#
# PURPOSE:
#
# 1. Debug OCR issues
# 2. Audit extraction process
# 3. Reuse extracted text without rerunning OCR
#
# FUTURE IMPROVEMENT:
# Save page-level confidence scores.
# ============================================================

output_md_file = "raw_extracted_text.md"

with open(output_md_file, "w", encoding="utf-8") as f:
    f.write(raw_text)

print(f"Raw OCR output saved to {output_md_file}")


# ============================================================
# STEP 7: INITIALIZE GEMINI CLIENT
# ============================================================
#
# PURPOSE:
# Use Gemini to transform unstructured OCR text into
# structured business data.
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# STEP 8: LLM-BASED STRUCTURED EXTRACTION
# ============================================================
#
# PURPOSE:
# Convert raw OCR text into structured JSON.
#
# EXTRACTS:
# - Contract Information
# - Provider Information
# - Reimbursement Rules
# - Compliance Requirements
# - Contact Details
# - Signatory Information
#
# TECHNIQUE:
# Prompt Engineering + Schema Enforcement
#
# FUTURE IMPROVEMENTS:
# - Few-shot prompting
# - Fine-tuned extraction model
# - RAG-based extraction
# ============================================================

def intelligent_parse(raw_text):

    if not raw_text.strip():
        return {}

    prompt = f"""
You are a STRICT JSON extraction engine.

CRITICAL INSTRUCTIONS:
- Output ONLY valid JSON.
- No markdown.
- No explanations.
- No comments.
- No hallucinated values.
- Missing values should be null.

DOCUMENT TEXT:
{raw_text}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not response.candidates:
            logging.error("No Gemini response.")
            return {}

        text = response.candidates[0].content.parts[0].text.strip()

        # ====================================================
        # SAFETY NET
        # ====================================================
        #
        # Sometimes LLMs add extra explanations.
        # Extract only the JSON object.
        #
        # FUTURE IMPROVEMENT:
        # Pydantic schema validation
        # ====================================================

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if not match:
            logging.error("No valid JSON found.")
            return {}

        return json.loads(match.group(0))

    except Exception as e:

        logging.error(
            f"Structured Extraction Failed: {e}"
        )

        return {}


# ============================================================
# STEP 9: POST-PROCESSING & FALLBACK EXTRACTION
# ============================================================
#
# PURPOSE:
#
# LLMs occasionally miss fields.
#
# This layer acts as a backup mechanism.
#
# CURRENT FALLBACKS:
# - Contract Number
# - Effective Date
# - Email Address
#
# FUTURE IMPROVEMENTS:
#
# 1. NER-based fallback extraction
# 2. Section classifier
# 3. Knowledge-base validation
#
# BENEFIT:
# Higher extraction accuracy.
# ============================================================

def post_process_critical_fields(result, raw_text):

    # --------------------------------------------------------
    # Contract Number Fallback
    # --------------------------------------------------------

    if not result["contract_summary"]["contract_number"]:

        match = re.search(
            r"ICMPT[_\s]*Ancillary[_\s]*\d+",
            raw_text,
            re.IGNORECASE
        )

        if match:
            result["contract_summary"]["contract_number"] = (
                match.group(0).replace(" ", "")
            )

    # --------------------------------------------------------
    # Effective Date Fallback
    # --------------------------------------------------------

    if not result["contract_summary"]["effective_date"]:

        date_match = re.search(
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            raw_text
        )

        if date_match:
            result["contract_summary"]["effective_date"] = (
                date_match.group(0)
            )

    # --------------------------------------------------------
    # Email Fallback
    # --------------------------------------------------------

    if not result["participation_agreement_cover"]["contract_contact_email"]:

        email_match = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            raw_text
        )

        if email_match:
            result["participation_agreement_cover"][
                "contract_contact_email"
            ] = email_match.group(0)

    return result


# ============================================================
# STEP 10: HYBRID EXTRACTION STRATEGY
# ============================================================
#
# CURRENT APPROACH:
#
# Primary:
# Gemini Extraction
#
# Secondary:
# Regex Fallback Extraction
#
# BENEFIT:
# More reliable than using only one technique.
#
# FUTURE IMPROVEMENTS:
#
# LLM
#  ↓
# NER
#  ↓
# Regex
#  ↓
# Human Validation
#
# ============================================================

result = intelligent_parse(raw_text)

result = post_process_critical_fields(
    result,
    raw_text
)

print(json.dumps(result, indent=2))


# ============================================================
# STEP 11: EXPORT CONTRACT SUMMARY
# ============================================================

df_contract = pd.DataFrame(
    [result["contract_summary"]]
)

df_contract.to_csv(
    "contract_summary.csv",
    index=False
)


# ============================================================
# STEP 12: EXPORT PROVIDER DETAILS
# ============================================================

df_provider = pd.DataFrame(
    [result["provider_details"]]
)

df_provider.to_csv(
    "provider_details.csv",
    index=False
)


# ============================================================
# STEP 13: EXPORT COMPLIANCE REQUIREMENTS
# ============================================================

df_compliance = pd.DataFrame(
    result["compliance_requirements"],
    columns=["requirement"]
)

df_compliance.to_csv(
    "compliance_requirements.csv",
    index=False
)


# ============================================================
# STEP 14: EXPORT REIMBURSEMENT RULES
# ============================================================

df_reimbursement = pd.DataFrame(
    result["reimbursement_rules"]
)

df_reimbursement.to_csv(
    "reimbursement_rules.csv",
    index=False
)


# ============================================================
# STEP 15: FLATTEN NESTED REIMBURSEMENT DATA
# ============================================================
#
# PURPOSE:
#
# Convert nested reimbursement rules into
# analytics-friendly tabular rows.
#
# Example:
#
# Contract A | Rule 1
# Contract A | Rule 2
# Contract A | Rule 3
#
# BENEFIT:
#
# Easier reporting
# Easier dashboarding
# Easier database loading
#
# FUTURE IMPROVEMENTS:
#
# - Rule confidence score
# - Duplicate detection
# - Rule categorization
# ============================================================

rows = []

base_fields = {
    **result["contract_summary"],
    **result["provider_details"],
    **result["contact_information"],
    **result["signatory_details"]
}

for rule in result["reimbursement_rules"]:

    row = base_fields.copy()

    row.update(rule)

    rows.append(row)

df_reimbursement_expanded = pd.DataFrame(rows)

df_reimbursement_expanded.to_csv(
    "reimbursement_expanded.csv",
    index=False
)

print("Pipeline completed successfully.")

# ============================================================
# POSSIBLE ENHANCEMENTS FOR DISCUSSION WITH SENIOR ENGINEERS
# ============================================================
#
# 1. OCR Fallback Mechanism
#    PaddleOCR → Tesseract
#
# 2. Confidence-Based Reprocessing
#
# 3. Dynamic Image Preprocessing
#
# 4. RAG-Based Contract Extraction
#
# 5. Named Entity Recognition (NER)
#
# 6. Human-In-The-Loop Validation
#
# 7. Duplicate Detection Layer
#
# 8. Pydantic Schema Validation
#
# 9. Monitoring & Logging Dashboard
#
# 10. Fine-Tuned Domain-Specific Model
#
# ============================================================