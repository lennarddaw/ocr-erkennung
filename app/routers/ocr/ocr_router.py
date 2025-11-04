from fastapi import APIRouter, HTTPException
from app.routers.ocr.model import ImageBase64
import base64
import re
from PIL import Image
from io import BytesIO
import pytesseract
import cv2
import numpy as np
from pathlib import Path
import json
from rapidfuzz import fuzz, process
from pillow_heif import register_heif_opener

register_heif_opener()

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

CURRENT_DIR = Path(__file__).parent
RECIPIENT_FILE = CURRENT_DIR / "recipients.json"

def load_recipients():
    try:
        with open(RECIPIENT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("recipients", []), data.get("settings", {})
    except FileNotFoundError:
        print(f"{RECIPIENT_FILE} nicht gefunden")
        return [], {}
    except Exception as e:
        print(f"Fehler beim Laden der Empfänger: {str(e)}")
        return [], {}
    
KNOWN_RECIPIENTS, SETTINGS = load_recipients()
FUZZY_THRESHOLD = SETTINGS.get("fuzzy_threshold", 70)
MIN_WORD_LENGTH = SETTINGS.get("min_word_length", 2)
ENABLE_FALLBACK = SETTINGS.get("enable_fallback", True)

def convert_heic_to_rgb(pil_image):
    try:
        if pil_image.format == 'HEIF' or pil_image.format == 'HEIC':
            pil_image = pil_image.convert('RGB')
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        return pil_image
    except Exception as e:
        print(f"Warnung bei HEIC-Konvertierung: {e}")
        return pil_image.convert('RGB')

# auch mit preprocessing funktioniert es nicht, Name wird aber erkannt
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
    # noch ein wenig Rauschreduzierung
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)

    return cleaned


def extract_name_candidates(text: str) -> list:
    if not text or not text.strip():
        return []
    
    candidates = []

    for line in text.split('\n'):
        clean = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', line)

        words = clean.split()

        for i in range(len(words) -1):
            word1 = words[i].strip()
            word2 = words[i + 1].strip()

            if (len(word1) >= MIN_WORD_LENGTH and
                len(word2) >= MIN_WORD_LENGTH and
                word1[0].isupper() and 
                word2[0].isupper()):

                candidate = f"{word1} {word2}"
                candidates.append(candidate)

    return candidates

def find_best_recipient_match(candidates: list) -> dict:
    if not candidates:
        return {
            "name": "Kein Kandidat gefunden",
            "confidence": 0,
            "ocr_text": None,
            "method": "none"
        }
    
    if not KNOWN_RECIPIENTS:
        return {
            "name": candidates[0],
            "confidence": 50,
            "ocr_text": candidates[0],
            "method": "no_recipient_list",
            "warning": "Keine Empfängerliste geladen"
        }
    
    best_match = None
    best_score = 0
    best_candidate = None
    
    for candidate in candidates:
        match = process.extractOne(
            candidate,
            KNOWN_RECIPIENTS,
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] > best_score:
            best_score = match[1]
            best_match = match[0]
            best_candidate = candidate
    
    if best_match and best_score >= FUZZY_THRESHOLD:
        return {
            "name": best_match,
            "confidence": best_score,
            "ocr_text": best_candidate,
            "method": "fuzzy_match",
            "all_candidates": candidates
        }
    elif ENABLE_FALLBACK and candidates:
        return {
            "name": best_candidate if best_candidate else candidates[0],
            "confidence": best_score,
            "ocr_text": best_candidate if best_candidate else candidates[0],
            "method": "fallback_low_confidence",
            "all_candidates": candidates,
            "warning": f"Niedriger Confidence Score ({best_score}). Möglicher Match: {best_match}"
        }
    else:
        return {
            "name": "Keinen passenden Empfänger gefunden",
            "confidence": 0,
            "ocr_text": best_candidate,
            "method": "no_match",
            "all_candidates": candidates
        }



def extract_recipient(text: str) -> dict:
    candidates = extract_name_candidates(text)

    result = find_best_recipient_match(candidates)

    return result



@router.post("/upload-image")
async def upload_image(image_data: ImageBase64):
    # upload läuft aber clean durch
    try:
        img_bytes = base64.b64decode(image_data.img_body_base64)

        pil_image = Image.open(BytesIO(img_bytes))
        pil_image = convert_heic_to_rgb(pil_image)

        numpy_image = np.array(pil_image)

        preprocessed_image = preprocess_image(numpy_image)

        custom_config = r'--oem 3 --psm 6'
        ocr_text = pytesseract.image_to_string(
            preprocessed_image, 
            lang='deu',
            config=custom_config
        )

        recipient_result = extract_recipient(ocr_text)

        return {
            "success": True,
            "message": "Bild erfolgreich empfangen",
            "image_size": len(img_bytes),
            "ocr_text": ocr_text,
            "recipient": recipient_result["name"],
            "recipient_details": recipient_result,
            "image_base64": image_data.img_body_base64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fehler beim Verarbeiten des Bildes: {str(e)}")
    
# neue get/ für die Empfänger
@router.get("/recipients")
async def get_recipients():
    return {
        "recipients": KNOWN_RECIPIENTS,
        "count": len(KNOWN_RECIPIENTS),
        "settings": SETTINGS
    }

@router.post("/reload-recipients")
async def reload_recipients():

    global KNOWN_RECIPIENTS, SETTINGS, FUZZY_THRESHOLD, MIN_WORD_LENGTH, ENABLE_FALLBACK
    
    KNOWN_RECIPIENTS, SETTINGS = load_recipients()
    FUZZY_THRESHOLD = SETTINGS.get("fuzzy_threshold", 70)
    MIN_WORD_LENGTH = SETTINGS.get("min_word_length", 2)
    ENABLE_FALLBACK = SETTINGS.get("enable_fallback", True)
    
    return {
        "success": True,
        "message": "Empfängerliste neu geladen",
        "recipients_count": len(KNOWN_RECIPIENTS),
        "settings": SETTINGS
    }