from fastapi import APIRouter, HTTPException
from app.routers.ocr.model import ImageBase64
import base64
import re
from PIL import Image
from io import BytesIO
import pytesseract
import cv2
import numpy as np 

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

def preprocess_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scaled = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return scaled


def extract_recipient(text: str) -> str:
    # patterns alleine reicht nicht aus
    patterns = [
        r'(?:An|an):\s*(.+?)(?:\n|$)',
        r'(?:Empfänger|empfänger):\s*(.+?)(?:\n|$)',
        r'(?:To|to):\s*(.+?)(?:\n|$)',
        r'(?:Herrn|Frau|Firma)\s+(.+?)(?:\n|$)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            recipient = match.group(1).strip()
            recipient = re.sub(r'\s+', ' ', recipient)
            return recipient
        
    return "Keinen Empfänger gefunden"



@router.post("/upload-image")
async def upload_image(image_data: ImageBase64):
    # upload läuft aber clean durch
    try:
        img_bytes = base64.b64decode(image_data.img_body_base64)

        pil_image = Image.open(BytesIO(img_bytes))
        numpy_image = np.array(pil_image)

        preprocessed_image = preprocess_image(numpy_image)

        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(
            preprocessed_image, 
            lang='deu+eng',
            config=custom_config
        )
        
        recipient = extract_recipient(ocr_text)
        
        return {
            "success": True,
            "message": "Bild erfolgreich empfangen",
            "image_size": len(img_bytes),
            "ocr_text": ocr_text,
            "recipient": recipient,
            "image_base64": image_data.img_body_base64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fehler beim Verarbeiten des Bildes: {str(e)}")