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
# MATCH THRESHOLDS
# ============================================================

KEYWORD_THRESHOLD    = 80
FINAL_SCORE_THRESHOLD = 65

LANGUAGES = {

    "IFT": {
        "target_text": """
            In the case of a transfer between facilities within a xx hour period, 
            payment will be adjusted accordingly. Such transfers and RAs will not 
            be treated as a separate discharge and admission.
        """,
        "keywords": [
            "transfer between facilities",
            "xx hour period",
            "payment will be adjusted",
            "transfers and RAs",
            "separate discharge and admission",
            "treated as a separate discharge"
        ]
    },

    "RA": {
        "target_text": """
            An RA on the same date or within xx hours for the same condition 
            shall be treated as a single admission, per applicable payment guidelines. 
            For hospice patients, prior days will follow the patient if the RA occurs within xx days; 
            otherwise treated as a new election.
        """,
        "keywords": [
            "RA on the same date",
            "within xx hours",
            "same condition",
            "single admission",
            "applicable payment guidelines",
            "hospice patients",
            "RA within xx days",
            "new election"
        ]
    },

    # "outlier payment": {
    #     "target_text": """
    #         Outlier payments shall be made when the covered charges exceed 
    #         the outlier threshold as defined by Medicare guidelines.
    #     """,
    #     "keywords": [
    #         "outlier payment",
    #         "outlier threshold",
    #         "covered charges exceed",
    #         "medicare guidelines",
    #     ]
    # }
    
}

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

    if text.isupper():
        return True

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

def keyword_match_score(text, keywords):

    scores = []

    for kw in keywords:

        score = fuzz.partial_ratio(
            kw.lower(),
            text.lower()
        )

        scores.append(score)

    return max(scores)


# ============================================================
# TARGET TEXT SIMILARITY
# ============================================================

def target_text_similarity(text, target_text):

    score = fuzz.token_set_ratio(
        target_text.lower(),
        text.lower()
    )

    return score


# ============================================================
# FINAL SCORE
# ============================================================

def compute_final_score(text, target_text, keywords):

    keyword_score = keyword_match_score(text, keywords)

    if keyword_score < 45:
        return 0

    text_similarity = target_text_similarity(text, target_text)

    final_score = (
        keyword_score * 0.4 +
        text_similarity * 0.6
    )

    return final_score


# ============================================================
# REMOVE NOISE
# ============================================================

def is_noise(text):

    if len(text) < 25:
        return True

    numeric_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)

    if numeric_ratio > 0.4:
        return True

    if re.search(r'\.{5,}', text):
        return True

    return False


# ============================================================
# SENTENCE LEVEL PRE-CHECK
# ============================================================

def block_has_relevant_sentence(text, target_text, keywords):

    sentences = re.split(r'(?<=[.!?])\s+|(?<=:)\s+|(?<=;)\s+', text)

    for sentence in sentences:

        s = sentence.strip()

        if not s:
            continue

        kw_score = keyword_match_score(s, keywords)
        txt_score = target_text_similarity(s, target_text)
        combined = (kw_score * 0.4) + (txt_score * 0.6)

        if combined >= 60:
            return True

    return False


# ============================================================
# TRIM IRRELEVANT CONTENTS
# ============================================================

def trim_irrelevant_contents(text, target_text, keywords):

    sentences = re.split(r'(?<=[.!?])\s+|(?<=:)\s+|(?<=;)\s+', text)

    if not sentences:
        return text

    scored = []

    for i, sentence in enumerate(sentences):

        s = sentence.strip()

        if not s:
            continue

        kw_score = keyword_match_score(s, keywords)
        txt_score = target_text_similarity(s, target_text)
        combined = (kw_score * 0.4) + (txt_score * 0.6)

        scored.append((i, s, combined))

    if not scored:
        return text

    HIGH_SCORE_THRESHOLD = 60

    first_idx = None
    last_idx  = None

    for i, s, score in scored:
        if score >= HIGH_SCORE_THRESHOLD:
            if first_idx is None:
                first_idx = i
            last_idx = i

    if first_idx is None:
        best = max(scored, key=lambda x: x[2])
        return best[1]

    relevant = [
        s for i, s, score in scored
        if first_idx <= i <= last_idx
    ]

    return " ".join(relevant)


# ============================================================
# HARD FILTER — TIMEFRAME REQUIRED
# ============================================================

def extract_timeframe(text):

    text_lower = text.lower()

    # -------------------------------------------------------
    # SAME DAY — check first before hour patterns
    # -------------------------------------------------------
    same_day_patterns = [
        "same day",
        "same-day",
        "same date",
        "same-date",
        "admitted the same day",
        "discharged and admitted the same day",
    ]

    for pattern in same_day_patterns:
        if pattern in text_lower:
            return "same day"

    # -------------------------------------------------------
    # GENERIC HOUR PATTERN — any number + hour(s)
    # handles: 24 hour, (24) hour, 48 hours, 72-hour
    # -------------------------------------------------------
    hour_match = re.search(
        r'\(?\s*(\d+)\s*\)?\s*[-]?\s*hour',
        text_lower
    )

    if hour_match:
        return f"{hour_match.group(1)} hour"

    # -------------------------------------------------------
    # WRITTEN NUMBER + HOUR
    # e.g. twenty-four hour, forty-eight hours
    # -------------------------------------------------------
    written_match = re.search(
        r'(twenty[\-\s]four|forty[\-\s]eight|seventy[\-\s]two)\s*[-]?\s*hour',
        text_lower
    )

    if written_match:
        word_to_num = {
            "twenty-four": "24",
            "twenty four": "24",
            "forty-eight": "48",
            "forty eight": "48",
            "seventy-two": "72",
            "seventy two": "72",
        }
        word = written_match.group(1)
        num  = word_to_num.get(word, word)
        return f"{num} hour"

    # -------------------------------------------------------
    # GENERIC DAY PATTERN — any number + day(s)
    # handles: 60 days, 30 days
    # -------------------------------------------------------
    day_match = re.search(
        r'\(?\s*(\d+)\s*\)?\s*[-]?\s*days?',
        text_lower
    )

    if day_match:
        return f"{day_match.group(1)} day"

    return None


# ============================================================
# EXTRACT TEXT FROM PAGE (native + OCR fallback)
# ============================================================

def extract_page_text(page):

    text = page.get_text().strip()

    if text:
        return text, "native"

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

def extract_text_from_pdf(pdf_path, languages_config):

    results = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"\n  Processing: {os.path.basename(pdf_path)}")
    print(f"  Pages: {total_pages}")

    # Build a single full text string with page mapping
    full_text = ""
    page_map = []  # stores (start_char_idx, end_char_idx, page_no)

    for page_num in range(total_pages):
        try:
            page = doc[page_num]
            text, method = extract_page_text(page)
            if not text:
                continue

            start_idx = len(full_text)
            full_text += text + "\n\n"
            end_idx = len(full_text)
            page_map.append((start_idx, end_idx, page_num + 1))
        except Exception as e:
            print(f"  Error page {page_num+1}: {e}")

    doc.close()

    if not full_text.strip():
        return results

    blocks = []
    # We will split manually and keep track of character offsets to figure out the page number
    for match in re.finditer(r'(?:^|\n\s*\n)(.+?)(?=\n\s*\n|$)', full_text, flags=re.DOTALL):
        raw_blk = match.group(1)
        start_char = match.start(1)
        end_char = match.end(1)
        
        blk = clean_text(raw_blk)
        if len(blk) > 25:
            blocks.append((blk, start_char, end_char))

    # We track section_mode per language using a dictionary
    section_modes = {lang: False for lang in languages_config.keys()}

    for blk, start_char, end_char in blocks:
        if is_noise(blk):
            continue

        best_lang = None
        best_score = 0
        best_trimmed = None
        best_timeframe = None

        for lang_ind, lang_config in languages_config.items():
            target_text = lang_config["target_text"]
            keywords = lang_config["keywords"]

            # ------------------------------------------------
            # SECTION DETECTION
            # ------------------------------------------------
            heading_score = keyword_match_score(blk, keywords)
            if is_heading(blk):
                if heading_score >= KEYWORD_THRESHOLD:
                    section_modes[lang_ind] = True
                else:
                    section_modes[lang_ind] = False

            # ------------------------------------------------
            # FINAL SCORING
            # ------------------------------------------------
            final_score = compute_final_score(blk, target_text, keywords)
            
            if section_modes[lang_ind]:
                final_score += 8

            if block_has_relevant_sentence(blk, target_text, keywords):
                final_score += 10

            # ------------------------------------------------
            # FILTER
            # ------------------------------------------------
            if final_score >= FINAL_SCORE_THRESHOLD:
                trimmed_text = trim_irrelevant_contents(blk, target_text, keywords)
                timeframe = extract_timeframe(trimmed_text)

                if final_score > best_score:
                    best_score = final_score
                    best_lang = lang_ind
                    best_trimmed = trimmed_text
                    best_timeframe = timeframe if timeframe else "Not specified"

        if best_lang:
            # Determine page number(s)
            matched_pages = []
            for p_start, p_end, p_num in page_map:
                if start_char < p_end and end_char > p_start:
                    matched_pages.append(str(p_num))
            
            page_str = ", ".join(matched_pages) if matched_pages else "Unknown"

            results.append({
                "lang_ind":    best_lang,
                "lang_desc":   best_trimmed,
                "page_no":     page_str,
                "score":       round(best_score, 2),
                "timeframe":   best_timeframe,
                "file_ext":    os.path.splitext(os.path.basename(pdf_path))[1],
                "source_file": os.path.basename(pdf_path)
            })

    return results


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    df["normalized"] = df["lang_desc"].str.lower()
    df["page_no"] = df["page_no"].astype(str)

    df = df.groupby(["source_file", "file_ext", "lang_ind", "normalized"], as_index=False).agg({
        "lang_desc": "first",
        "timeframe": "first",
        "page_no": lambda x: ", ".join(sorted(set(", ".join(x).split(", ")), key=lambda y: int(y) if y.isdigit() else y))
    })

    df = df.drop(columns=["normalized"])

    return df


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_documents():

    all_results = []

    os.makedirs("output", exist_ok=True)

    # --------------------------------------------------------
    # MANUAL FILE INPUT
    # --------------------------------------------------------

    print("\nEnter PDF file paths one by one.")
    print("Press ENTER with no input when done.\n")

    pdf_files = []

    while True:

        path = input("Enter PDF path: ").strip().strip('"')

        if not path:
            break

        if not os.path.isfile(path):
            print(f"  File not found: {path} — skipping.")
            continue

        if not path.lower().endswith(".pdf"):
            print(f"  Not a PDF file: {path} — skipping.")
            continue

        pdf_files.append(path)
        print(f"  Added: {path}")

    if not pdf_files:
        print("\nNo valid PDF files provided. Exiting.")
        return

    print(f"\nPDF files to process: {len(pdf_files)}")
    print(f"Languages to extract: {list(LANGUAGES.keys())}\n")

    # --------------------------------------------------------
    # PROCESS EACH PDF
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        text_results = extract_text_from_pdf(
            pdf_path,
            LANGUAGES
        )

        all_results.extend(text_results)

    # --------------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------------

    if not all_results:
        print("\nNo relevant content found.")
        return

    df = pd.DataFrame(all_results)

    df = remove_duplicates(df)

    df = df.sort_values(by=["source_file", "lang_ind", "page_no"])

    df = df[
        [
            "lang_ind",
            "lang_desc",
            "page_no",
            "source_file",
            "file_ext",
            "timeframe"
        ]
    ]

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nExtraction completed.")
    print(f"Output saved:      {OUTPUT_CSV}")
    print(f"Records extracted: {len(df)}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    process_documents()
