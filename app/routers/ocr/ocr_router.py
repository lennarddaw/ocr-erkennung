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
from typing import List, Dict, Tuple, Optional

register_heif_opener()

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

CURRENT_DIR = Path(__file__).parent
RECIPIENTS_FILE = CURRENT_DIR / "recipients.json"
LOCATIONS_FILE = CURRENT_DIR / "locations.json"


def load_recipients() -> Tuple[List[str], Dict]:
    try:
        with open(RECIPIENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            recipients = data.get("recipients", [])
            settings = data.get("settings", {})
            print(f"{len(recipients)} Empfänger geladen")
            return recipients, settings
    except FileNotFoundError:
        print(f"Warnung: {RECIPIENTS_FILE} nicht gefunden")
        return [], {}
    except json.JSONDecodeError as e:
        print(f"JSON-Fehler in recipients.json: {e}")
        return [], {}
    except Exception as e:
        print(f"Fehler beim Laden der Empfänger: {e}")
        return [], {}


def load_locations() -> Tuple[List[str], List[str], Dict]:
    try:
        with open(LOCATIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            locations = data.get("locations", [])
            keywords = data.get("company_keywords", [])
            settings = data.get("settings", {})
            print(f"✓ {len(locations)} Standorte und {len(keywords)} Keywords geladen")
            return locations, keywords, settings
    except FileNotFoundError:
        print(f"⚠ Warnung: {LOCATIONS_FILE} nicht gefunden - Filter deaktiviert")
        return [], [], {"filter_enabled": False}
    except json.JSONDecodeError as e:
        print(f"✗ JSON-Fehler in locations.json: {e}")
        return [], [], {"filter_enabled": False}
    except Exception as e:
        print(f"✗ Fehler beim Laden der Standorte: {e}")
        return [], [], {"filter_enabled": False}


KNOWN_RECIPIENTS, RECIPIENT_SETTINGS = load_recipients()
KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS = load_locations()

FUZZY_THRESHOLD = RECIPIENT_SETTINGS.get("fuzzy_threshold", 70)
MIN_WORD_LENGTH = RECIPIENT_SETTINGS.get("min_word_length", 2)
ENABLE_FALLBACK = RECIPIENT_SETTINGS.get("enable_fallback", True)

LOCATION_FUZZY_THRESHOLD = LOCATION_SETTINGS.get("location_fuzzy_threshold", 85)
FILTER_ENABLED = LOCATION_SETTINGS.get("filter_enabled", True)
CASE_SENSITIVE = LOCATION_SETTINGS.get("case_sensitive", False)


def convert_heic_to_rgb(pil_image: Image.Image) -> Image.Image:
    try:
        if pil_image.format in ['HEIF', 'HEIC']:
            print("ℹ HEIC-Format erkannt, konvertiere zu RGB...")
            pil_image = pil_image.convert('RGB')
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        return pil_image
    except Exception as e:
        print(f"⚠ Warnung bei HEIC-Konvertierung: {e}")
        return pil_image.convert('RGB')


def preprocess_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scaled = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)

    return cleaned


def is_location_or_company(candidate: str) -> Tuple[bool, Optional[str]]:
    if not FILTER_ENABLED:
        return False, None
    
    if KNOWN_LOCATIONS:
        match = process.extractOne(
            candidate,
            KNOWN_LOCATIONS,
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] >= LOCATION_FUZZY_THRESHOLD:
            return True, f"Standort erkannt: {match[0]} (Score: {match[1]})"
    
    for keyword in COMPANY_KEYWORDS:
        if CASE_SENSITIVE:
            if keyword in candidate:
                return True, f"Firmen-Keyword gefunden: {keyword}"
        else:
            if keyword.lower() in candidate.lower():
                return True, f"Firmen-Keyword gefunden: {keyword}"
    
    return False, None


def extract_name_candidates(text: str) -> List[Dict]:
    if not text or not text.strip():
        return []
    
    candidates = []
    
    for line_num, line in enumerate(text.split('\n'), 1):
        clean = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', line)
        
        words = clean.split()
        
        for i in range(len(words) - 1):
            word1 = words[i].strip()
            word2 = words[i + 1].strip()
            
            if (len(word1) >= MIN_WORD_LENGTH and 
                len(word2) >= MIN_WORD_LENGTH and
                word1[0].isupper() and 
                word2[0].isupper()):
                
                candidate = f"{word1} {word2}"

                is_location, reason = is_location_or_company(candidate)
                
                candidates.append({
                    "name": candidate,
                    "line_number": line_num,
                    "is_location": is_location,
                    "filter_reason": reason
                })
    
    return candidates


def filter_location_candidates(candidates: List[Dict]) -> List[str]:
    filtered = []
    filtered_out = []
    
    for candidate in candidates:
        if candidate["is_location"]:
            filtered_out.append({
                "name": candidate["name"],
                "reason": candidate["filter_reason"]
            })
        else:
            filtered.append(candidate["name"])
    
    if filtered_out:
        print(f"ℹ {len(filtered_out)} Kandidaten gefiltert:")
        for item in filtered_out:
            print(f"  - {item['name']}: {item['reason']}")
    
    return filtered


def find_best_recipient_match(candidates: List[str]) -> Dict:
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


def extract_recipient(text: str) -> Dict:

    all_candidates = extract_name_candidates(text)
    
    filtered_candidates = filter_location_candidates(all_candidates)
    
    result = find_best_recipient_match(filtered_candidates)
    
    result["total_candidates_found"] = len(all_candidates)
    result["filtered_candidates"] = len(all_candidates) - len(filtered_candidates)
    
    return result


@router.post("/upload-image")
async def upload_image(image_data: ImageBase64):
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
            "message": "Bild erfolgreich verarbeitet",
            "image_size": len(img_bytes),
            "image_format": pil_image.format or "Unknown",
            "ocr_text": ocr_text,
            "recipient": recipient_result["name"],
            "recipient_details": recipient_result,
            "image_base64": image_data.img_body_base64
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Fehler beim Verarbeiten des Bildes: {str(e)}"
        )


@router.get("/recipients")
async def get_recipients():
    return {
        "recipients": KNOWN_RECIPIENTS,
        "count": len(KNOWN_RECIPIENTS),
        "settings": RECIPIENT_SETTINGS
    }


@router.get("/locations")
async def get_locations():
    return {
        "locations": KNOWN_LOCATIONS,
        "keywords": COMPANY_KEYWORDS,
        "location_count": len(KNOWN_LOCATIONS),
        "keyword_count": len(COMPANY_KEYWORDS),
        "settings": LOCATION_SETTINGS
    }

@router.post("/reload-recipients")
async def reload_recipients():
    global KNOWN_RECIPIENTS, RECIPIENT_SETTINGS, FUZZY_THRESHOLD, MIN_WORD_LENGTH, ENABLE_FALLBACK
    
    KNOWN_RECIPIENTS, RECIPIENT_SETTINGS = load_recipients()
    FUZZY_THRESHOLD = RECIPIENT_SETTINGS.get("fuzzy_threshold", 70)
    MIN_WORD_LENGTH = RECIPIENT_SETTINGS.get("min_word_length", 2)
    ENABLE_FALLBACK = RECIPIENT_SETTINGS.get("enable_fallback", True)
    
    return {
        "success": True,
        "message": "Empfängerliste neu geladen",
        "recipients_count": len(KNOWN_RECIPIENTS),
        "settings": RECIPIENT_SETTINGS
    }


@router.post("/reload-locations")
async def reload_locations():
    global KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS
    global LOCATION_FUZZY_THRESHOLD, FILTER_ENABLED, CASE_SENSITIVE
    
    KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS = load_locations()
    LOCATION_FUZZY_THRESHOLD = LOCATION_SETTINGS.get("location_fuzzy_threshold", 85)
    FILTER_ENABLED = LOCATION_SETTINGS.get("filter_enabled", True)
    CASE_SENSITIVE = LOCATION_SETTINGS.get("case_sensitive", False)
    
    return {
        "success": True,
        "message": "Standortliste neu geladen",
        "locations_count": len(KNOWN_LOCATIONS),
        "keywords_count": len(COMPANY_KEYWORDS),
        "settings": LOCATION_SETTINGS
    }


@router.post("/reload-all")
async def reload_all():
    recipients_result = await reload_recipients()
    locations_result = await reload_locations()
    
    return {
        "success": True,
        "message": "Alle Listen neu geladen",
        "recipients": recipients_result,
        "locations": locations_result
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "recipients_loaded": len(KNOWN_RECIPIENTS) > 0,
        "locations_loaded": len(KNOWN_LOCATIONS) > 0,
        "filter_active": FILTER_ENABLED,
        "recipients_count": len(KNOWN_RECIPIENTS),
        "locations_count": len(KNOWN_LOCATIONS),
        "keywords_count": len(COMPANY_KEYWORDS)
    }