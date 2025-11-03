from fastapi import APIRouter, HTTPException
from app.routers.ocr.model import ImageBase64
import base64
import re
from PIL import Image
from io import BytesIO
import pytesseract

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

def extract_recipient(text: str) -> str:

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
    try:
        img_bytes = base64.b64decode(image_data.img_body_base64)

        image = Image.open(BytesIO(img_bytes))

        ocr_text = pytesseract.image_to_string(image, lang='deu+eng')
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