from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io
import re
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Paketerkennung",
    description="System zur Erkennung von Paketempfängern",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


def extract_recipient_name(text: str) -> Optional[str]:

    text = text.strip()
    
    patterns = [
        r'(?:An|an|AN):\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+)',
        r'(?:Für|für|FÜR):\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+)',
        r'(?:Empfänger|empfänger|EMPFÄNGER):\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+)',
        r'(?:To|TO):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:Recipient|RECIPIENT):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'^([A-ZÄÖÜ]{2,}\s+[A-ZÄÖÜ]{2,})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            name = match.group(1).strip()
            logger.info(f"Empfänger gefunden mit Pattern '{pattern}': {name}")
            return name
    
    lines = text.split('\n')
    for line in lines:
        words = line.strip().split()
        for i in range(len(words) - 1):
            if (words[i][0].isupper() and len(words[i]) > 2 and 
                words[i+1][0].isupper() and len(words[i+1]) > 2):
                name = f"{words[i]} {words[i+1]}"
                logger.info(f"Empfänger gefunden (Fallback): {name}")
                return name
    
    return None


def process_image_ocr(image: Image.Image) -> dict:
    try:
        text = pytesseract.image_to_string(
            image, 
            lang='deu+eng',
            config='--psm 6'
        )
        
        logger.info(f"OCR-Text erkannt (erste 200 Zeichen): {text[:200]}")
        
        recipient = extract_recipient_name(text)
        
        return {
            "success": True,
            "full_text": text,
            "recipient": recipient,
            "confidence": "high" if recipient else "low"
        }
    
    except Exception as e:
        logger.error(f"OCR Fehler: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "full_text": "",
            "recipient": None
        }


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Leitet zur Frontend-Seite um.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Dateityp nicht unterstützt. Erlaubt: {', '.join(allowed_types)}"
            )
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        file_path = UPLOAD_DIR / file.filename
        image.save(file_path)
        logger.info(f"Datei gespeichert: {file_path}")
        
        result = process_image_ocr(image)
        
        return JSONResponse(content={
            "filename": file.filename,
            "result": result
        })
    
    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """
    Health Check Endpoint
    """
    try:
        version = pytesseract.get_tesseract_version()
        return {
            "status": "healthy",
            "tesseract_version": str(version)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)