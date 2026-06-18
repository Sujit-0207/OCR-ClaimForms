import os
import re
import fitz  # PyMuPDF
import pandas as pd

from rapidfuzz import fuzz

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_CSV = "output/extracted_results_1.csv"

# Set tessdata path manually
TESSDATA_PATH = r"C:\Program Files\Tesseract-OCR\tessdata"

# ============================================================
# TARGET LANGUAGE / SECTION
# ============================================================

LANG_IND = "IFT"

# ------------------------------------------------------------
# PRIMARY MATCH TEXT
# ------------------------------------------------------------

TARGET_TEXT = """
In the case of a transfer between facilities within a xx hour period, 
payment will be adjusted accordingly. Such transfers and RAs will not 
be treated as a separate discharge and admission.
"""

# ------------------------------------------------------------
# RELATED KEYWORDS
# ------------------------------------------------------------

TARGET_KEYWORDS = [
    "transfer between facilities",
    "xx hour period",
    "payment will be adjusted",
    "transfers and RAs",
    "separate discharge and admission",
    "treated as a separate discharge"
]

# ============================================================
# MATCH THRESHOLDS
# ============================================================

KEYWORD_THRESHOLD = 80
TEXT_SIMILARITY_THRESHOLD = 72
FINAL_SCORE_THRESHOLD = 75

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.I)
    text = re.sub(r'\.{2,}', ' ', text)

    return text.strip()


# ============================================================
# DETECT HEADING
# ============================================================

def is_heading(text):

    text = text.strip()

    if len(text) > 120:
        return False

    # UPPERCASE headings
    if text.isupper():
        return True

    # Numbered headings
    if re.match(r'^\d+(\.\d+)*\s+', text):
        return True

    return False


# ============================================================
# SPLIT INTO BLOCKS
# ============================================================

def split_into_blocks(text):

    blocks = []

    raw_blocks = re.split(r'\n\s*\n', text)

    for blk in raw_blocks:

        blk = clean_text(blk)

        if len(blk) > 25:
            blocks.append(blk)

    return blocks


# ============================================================
# KEYWORD MATCH SCORE
# ============================================================

def keyword_match_score(text):

    scores = []

    for kw in TARGET_KEYWORDS:

        score = fuzz.partial_ratio(
            kw.lower(),
            text.lower()
        )

        scores.append(score)

    return max(scores)


# ============================================================
# TARGET TEXT SIMILARITY
# ============================================================

def target_text_similarity(text):

    score = fuzz.token_set_ratio(
        TARGET_TEXT.lower(),
        text.lower()
    )

    return score


# ============================================================
# FINAL SCORE
# ============================================================

def compute_final_score(text):

    keyword_score = keyword_match_score(text)

    # Fast reject
    if keyword_score < 45:
        return 0

    text_similarity = target_text_similarity(text)

    final_score = (
        keyword_score * 0.4 +
        text_similarity * 0.6
    )

    return final_score


# ============================================================
# REMOVE NOISE
# ============================================================

def is_noise(text):

    # Very short
    if len(text) < 25:
        return True

    # Mostly numeric
    numeric_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    if numeric_ratio > 0.4:
        return True

    # TOC patterns
    if re.search(r'\.{5,}', text):
        return True

    return False


# ============================================================
# TRIM IRRELEVANT CONTENTS
# ============================================================

def trim_irrelevant_contents(text):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    if not sentences:
        return text

    relevant_sentences = []

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue

        kw_score = keyword_match_score(s)
        txt_score = target_text_similarity(s)

        if kw_score >= 70 or txt_score >= 60:
            relevant_sentences.append(s)

    if relevant_sentences:
        return " ".join(relevant_sentences)

    return text


# ============================================================
# EXTRACT TEXT FROM PAGE (native + OCR fallback)
# ============================================================

def extract_page_text(page):

    # Try native text extraction first
    text = page.get_text().strip()

    if text:
        return text, "native"

    # If empty, fall back to OCR
    try:
        tp = page.get_textpage_ocr(
            flags=0,
            language="eng",
            dpi=300,
            tessdata=TESSDATA_PATH
        )
        text = page.get_text(textpage=tp).strip()
        return text, "ocr"

    except Exception as e:
        print(f"    OCR failed: {e}")
        return "", "failed"


# ============================================================
# EXTRACT TEXT CONTENT
# ============================================================

def extract_text_from_pdf(pdf_path):

    results = []

    doc = fitz.open(pdf_path)

    total_pages = len(doc)

    print(f"\nProcessing: {os.path.basename(pdf_path)}")
    print(f"Pages: {total_pages}")

    section_mode = False

    for page_num in range(total_pages):

        try:

            page = doc[page_num]

            # ------------------------------------------------
            # EXTRACT TEXT — native or OCR
            # ------------------------------------------------

            text, method = extract_page_text(page)

            print(f"  Page {page_num+1}: {method}")

            if not text:
                continue

            blocks = split_into_blocks(text)

            for blk in blocks:

                cleaned = clean_text(blk)

                if is_noise(cleaned):
                    continue

                # ------------------------------------------------
                # SECTION DETECTION
                # ------------------------------------------------

                heading_score = keyword_match_score(cleaned)

                if is_heading(cleaned):

                    if heading_score >= KEYWORD_THRESHOLD:
                        section_mode = True

                    else:
                        section_mode = False

                # ------------------------------------------------
                # FINAL SCORING
                # ------------------------------------------------

                final_score = compute_final_score(cleaned)

                # Section boost
                if section_mode:
                    final_score += 8

                # ------------------------------------------------
                # FILTER
                # ------------------------------------------------

                if final_score >= FINAL_SCORE_THRESHOLD:

                    trimmed_text = trim_irrelevant_contents(cleaned)

                    results.append({
                        "lang_ind": LANG_IND,
                        "lang_desc": trimmed_text,
                        "page_no": page_num + 1,
                        "score": round(final_score, 2),
                        "file_ext": os.path.splitext(os.path.basename(pdf_path))[1],
                        "source_file": os.path.basename(pdf_path)
                    })

        except Exception as e:
            print(f"  Error page {page_num+1}: {e}")

    doc.close()

    return results


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    df["normalized"] = df["lang_desc"].str.lower()

    df = df.drop_duplicates(
        subset=["normalized"]
    )

    df = df.drop(columns=["normalized"])

    return df


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_documents():

    all_results = []

    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------
    # FOLDER INPUT
    # --------------------------------------------------------

    PDF_FOLDER = r"C:\Users\rashmika\Desktop\Extractor\contract"   

    if not os.path.isdir(PDF_FOLDER):
        print(f"Folder not found: {PDF_FOLDER}")
        return

    pdf_files = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"No PDF files found in: {PDF_FOLDER}")
        return

    print(f"\nFound {len(pdf_files)} PDF files")

    # --------------------------------------------------------
    # PROCESS ALL PDFs
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        text_results = extract_text_from_pdf(pdf_path)

        all_results.extend(text_results)

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    if not all_results:
        print("\nNo relevant content found.")
        return

    df = pd.DataFrame(all_results)

    df = remove_duplicates(df)

    df = df.sort_values(
        by=["source_file", "page_no"]
    )

    df = df[
        [
            "lang_ind",
            "lang_desc",
            "page_no",
            "source_file",
            "file_ext"
        ]
    ]

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nExtraction completed.")
    print(f"Output saved: {OUTPUT_CSV}")
    print(f"Records extracted: {len(df)}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_documents()
