# HCFA 1500 OCR & LLM Extractor

The below linked Google Colabs provides all my works so far of claim forms;

**Google Colab** : https://colab.research.google.com/drive/1I0y7zI017VqJ8lhW57KOm5H94JMMf7mH?usp=sharing ; https://colab.research.google.com/drive/1453FGvr24eH4zJh56iFEhNr8TDro0Yl-?usp=sharing ; https://colab.research.google.com/drive/1CMQPNomhBafs2CUn07L6RUh-4VwFgQH7?usp=sharing ; https://colab.research.google.com/drive/1OSNU6rOoufzqoon51lkapJDpLVeBYNRF?usp=sharing 

##  Features
- **OCR Engine**: Utilizes Tesseract for high-quality text and layout extraction.
- **AI Extraction**: Uses GPT-4o to parse messy OCR output into structured JSON.
- **Preprocessing**: Includes image enhancement steps (grayscale, contrast, sharpening) to improve accuracy.

# Claim Form Extraction — CMS-1500 & UB-04

An automated tool for extracting structured data from scanned medical claim forms, removing
the need for manual data entry. It supports both **CMS-1500** (professional claims, used by
physicians and clinics) and **UB-04** (institutional claims, used by hospitals and
facilities), and automatically determines which form it is processing.

## Overview

For each document, the tool reads every field — patient and provider details, dates, charges,
diagnosis codes, and service or billing lines — and outputs the results in a structured
format. It additionally flags any values it could not read with confidence and performs a set
of validation checks, allowing reviewers to focus only on the items that require attention
rather than verifying every form.

## Usage

1. Provide your API key (Gemini) in the configuration section.
2. Place the forms to be processed (PDF or image) in the input folder.
3. Run the notebook cells in order.

The entire folder is processed in a single run.

## Output

The primary output is an **Excel workbook** containing the extracted data, including a
dedicated **Review** sheet that highlights items needing verification:

- **Yellow** — fields the tool flagged as low confidence (e.g., poor scan quality or
  handwriting).
- **Orange** — fields that failed a validation check, such as an invalid provider
  identifier, an unparseable date, or charges that do not reconcile with the stated total.

An empty Review sheet indicates the form was extracted cleanly.

## Notes

- Extraction accuracy depends on scan quality; clear, properly aligned scans produce the best
  results, and flagged fields should always be reviewed.
- The tool extracts data to files only — it does not submit, file, or transmit claims.
- Processing is fault-tolerant: a document that cannot be read is recorded in the run summary
  and does not interrupt the rest of the batch.)
