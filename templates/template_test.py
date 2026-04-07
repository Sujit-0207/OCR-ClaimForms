import cv2
import pytesseract
import json
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = r"data/claim forms/HCFA/1.png"
TEMPLATE_FILE = r"templates/template.json"
OUTPUT_FILE = r"templates/1_output.json"

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Could not read image from {IMAGE_PATH}")

b, g, r = cv2.split(img)
img_cleared = r 

STANDARD_W = 1200
STANDARD_H = 1600

def resize_with_padding(img):
    h, w = img.shape[:2]

    scale = min(STANDARD_W / w, STANDARD_H / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # Using 1-channel canvas since we're giving it the red-channel preprocessed image
    canvas = np.ones((STANDARD_H, STANDARD_W), dtype=np.uint8) * 255

    x_offset = (STANDARD_W - new_w) // 2
    y_offset = (STANDARD_H - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas, scale, x_offset, y_offset

img_processed, scale, x_off, y_off = resize_with_padding(img_cleared)

# Load template
with open(TEMPLATE_FILE, "r") as f:
    FIELDS = json.load(f)


def extract_text(crop):
    # Already grayscale (red channel), enhance it.
    # Scale up if the crop is small
    h_c, w_c = crop.shape
    if h_c < 30:
        crop = cv2.resize(crop, (w_c * 2, h_c * 2), interpolation=cv2.INTER_CUBIC)

    # Apply adaptive thresholding for robust binarization
    thresh = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 23, 15)

    # Minimal denoising
    denoised = cv2.medianBlur(thresh, 1)

    return pytesseract.image_to_string(
        denoised,
        config='--psm 7'
    ).strip()


output = {}

for field, box in FIELDS.items():
    x1, y1, x2, y2 = box

    crop = img_processed[y1:y2, x1:x2]
    text = extract_text(crop)

    output[field] = {
        "value": text,
        # "bbox": {
        #     "x1": x1,
        #     "y1": y1,
        #     "x2": x2,
        #     "y2": y2
        # }
    }

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print("✅ Extraction complete:", OUTPUT_FILE)