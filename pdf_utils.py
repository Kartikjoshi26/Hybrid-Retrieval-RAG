import fitz 
import random
import pytesseract
from PIL import Image
from io import BytesIO

def detect_pdf_type(pdf_path, sample_pages=5):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    sampled_indices = random.sample(range(total_pages), min(sample_pages, total_pages))

    normal_text, ocr_text = "", ""

    for i in sampled_indices:
        page = doc[i]

        normal_text += page.get_text()

        pix = page.get_pixmap(dpi=200)
        img = Image.open(BytesIO(pix.tobytes("png")))
        ocr_text += pytesseract.image_to_string(img)

    len_normal = len(normal_text.strip())
    len_ocr = len(ocr_text.strip())

    if len_normal == 0 and len_ocr > 0:
        pdf_type = "Scanned PDF"
    elif len_normal > 0 and len_ocr == 0:
        pdf_type = "Digital PDF"
    elif len_normal == len_ocr:
        pdf_type = "Digital PDF (OCR matches text)"
    elif len_normal > len_ocr:
        pdf_type = "Digital PDF (text dominates)"
    else:
        pdf_type = "Scanned PDF (OCR dominates)"

    return {
        "pdf_type": pdf_type
    }
