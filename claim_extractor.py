import cv2
import pytesseract
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# LOAD ENV & OPENAI CLIENT
# =========================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# OCR FUNCTION
# =========================
def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text


# =========================
# LLM EXTRACTION FUNCTION
# =========================
def extract_with_llm(ocr_text):
    prompt = f"""
You are a medical claims data extractor.

STRICT RULES:
- Output ONLY valid JSON
- No explanations
- No markdown
- No ```json
- Do NOT invent data
- Use null if a value is missing

JSON FORMAT:
{{
  "form_type": "CMS-1500",
  "patient_name": string | null,
  "insured_name": string | null,
  "diagnosis_codes": [string],
  "service_lines": [
    {{
      "cpt_code": string,
      "charges": string
    }}
  ]
}}

TEXT:
\"\"\"{ocr_text}\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured medical claim data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # ---- SAFE JSON EXTRACTION ----
    if content.startswith("```"):
        content = content.split("```")[1]

    start = content.find("{")
    end = content.rfind("}") + 1

    json_str = content[start:end]

    return json.loads(json_str)

def normalize_output(data):
    # ---- Normalize name format to CMS-1500 ----
    def to_last_first(name):
        if not name:
            return None
        parts = name.replace(",", "").split()
        if len(parts) >= 2:
            last = parts[-1]
            first = " ".join(parts[:-1])
            return f"{last}, {first}"
        return name

    data["patient_name"] = to_last_first(data.get("patient_name"))
    data["insured_name"] = to_last_first(data.get("insured_name"))

    # ---- Normalize ICD codes (add dot if missing) ----
    fixed_codes = []
    for code in data.get("diagnosis_codes", []):
        if len(code) > 3 and "." not in code:
            fixed_codes.append(code[:3] + "." + code[3:])
        else:
            fixed_codes.append(code)

    data["diagnosis_codes"] = fixed_codes

    # ---- Normalize charges (remove $) ----
    for svc in data.get("service_lines", []):
        if "charges" in svc and svc["charges"]:
            svc["charges"] = svc["charges"].replace("$", "")

    return data

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    image_path = "data/claim_form.png"
    os.makedirs("output", exist_ok=True)

    print("📸 Running OCR...")
    ocr_text = extract_text(image_path)

    print("🤖 Extracting using LLM...")
    raw_data = extract_with_llm(ocr_text)
    extracted_data = normalize_output(raw_data)


    with open("output/extracted_claim.json", "w") as f:
        json.dump(extracted_data, f, indent=4)

    print("✅ Extraction completed")
    print(json.dumps(extracted_data, indent=4))
