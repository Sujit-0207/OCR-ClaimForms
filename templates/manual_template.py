import cv2
import json

IMAGE_PATH = "data/claim forms/HCFA/1.png"
OUTPUT_TEMPLATE = "template.json"

import cv2
import numpy as np

STANDARD_W = 1200
STANDARD_H = 1600

def resize_with_padding(img):
    h, w = img.shape[:2]

    scale = min(STANDARD_W / w, STANDARD_H / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.ones((STANDARD_H, STANDARD_W, 3), dtype=np.uint8) * 255

    x_offset = (STANDARD_W - new_w) // 2
    y_offset = (STANDARD_H - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas, scale, x_offset, y_offset

img, scale, x_off, y_off = resize_with_padding(cv2.imread(IMAGE_PATH))

FIELD_NAMES = [
    "insured_id",   
    "patient_name",
    "patient_dob_month",
    "patient_dob_day",
    "patient_dob_year",
    "patient_gender_male",
    "patient_gender_female",
    "patient_city",
    "patient_state",
    "patient_zip",
    "patient_phone",
    "insured_name",
    "patient_address",
    "patient_relationship_to",
    "insured_address",
    "insured_city",
    "insured_state",
    "insured_zip",
    "insured_phone",
    "policy_number",
    "insurance_plan_name",
    "insured_dob_month",
    "insured_dob_day",
    "insured_dob_year",
    "insured_gender_male",
    "insured_gender_female",
    "patient_signature",
    "signature_date",
    "insured_signature",
    "ICD_indicator",
    "diagnosisCodes",
    "from_date_of_service",
    "to_date_of_service",
    "CPT/HCPCS_codes",
    "Modifiers",
    "diagnosis_pointers",
    "charges",
    "days_or_units",
    "NPI",
    "fedTex_id",
    "patient_ac_no",
    "accept_assignment",
    "total_charge",
    "amount_paid",
    "physician_signature",
    "signature_date",
    "service_facility_location"
]

rois = {}

drawing = False
ix, iy = -1, -1


def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = ix, iy
        x2, y2 = x, y

        rois[current_field] = [x1, y1, x2, y2]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, current_field, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"✅ {current_field} saved: {(x1,y1,x2,y2)}")


# Create a resizable window and set its initial size
cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selector", 800, 950) 
cv2.setMouseCallback("ROI Selector", draw_roi)

print("\n🚀 Guided ROI Selection Started")
print("👉 Draw box for each field in order\n")
print("💡 Use the mouse to draw boxes. The window is resizable if needed.")

for i, field in enumerate(FIELD_NAMES):

    current_field = field

    print(f"\n📌 [{i+1}/{len(FIELD_NAMES)}] Select ROI for: {field}")
    print("👉 Draw box using mouse")

    while True:
        cv2.imshow("ROI Selector", img)
        key = cv2.waitKey(1) & 0xFF

        # wait until ROI is drawn
        if field in rois:
            break

cv2.destroyAllWindows()

# Save template
with open(OUTPUT_TEMPLATE, "w") as f:
    json.dump(rois, f, indent=4)

print(f"\n✅ All {len(FIELD_NAMES)} ROIs captured successfully!")
print("📁 Saved to:", OUTPUT_TEMPLATE)
