from fastapi import APIRouter, HTTPException
from app.routers.ocr.model import ImageBase64
import base64
import re
from PIL import Image, ImageEnhance
from io import BytesIO
import pytesseract
import cv2
import numpy as np
from pathlib import Path
import json
from rapidfuzz import fuzz, process
from pillow_heif import register_heif_opener
from typing import List, Dict, Tuple, Optional, Set
import easyocr
import time
from itertools import combinations

register_heif_opener()

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

CURRENT_DIR = Path(__file__).parent
RECIPIENTS_FILE = CURRENT_DIR / "recipients.json"
LOCATIONS_FILE = CURRENT_DIR / "locations.json"

print("Initialisiere EasyOCR Reader...")
try:
    EASYOCR_READER = easyocr.Reader(['de', 'en'], gpu=False, verbose=False)
    print("EasyOCR Reader erfolgreich initialisiert")
except Exception as e:
    print(f"EasyOCR Fehler: {e}")
    EASYOCR_READER = None


def load_recipients() -> Tuple[List[str], Dict]:
    try:
        with open(RECIPIENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            recipients = data.get("recipients", [])
            settings = data.get("settings", {})
            print(f"✓ {len(recipients)} Empfänger geladen")
            return recipients, settings
    except FileNotFoundError:
        print(f"⚠ CRITICAL: {RECIPIENTS_FILE} nicht gefunden!")
        return [], {}
    except json.JSONDecodeError as e:
        print(f"✗ JSON-Fehler in recipients.json: {e}")
        return [], {}
    except Exception as e:
        print(f"✗ Fehler beim Laden der Empfänger: {e}")
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


def check_image_quality(image: np.ndarray) -> Dict:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    issues = []
    recommendations = []
    
    if brightness < 50:
        issues.append("Zu dunkel")
        recommendations.append("Bild bei besserem Licht aufnehmen")
    elif brightness > 200:
        issues.append("Zu hell/überbelichtet")
        recommendations.append("Belichtung reduzieren")
    
    if contrast < 30:
        issues.append("Geringer Kontrast")
        recommendations.append("Text deutlicher vom Hintergrund abheben")
    
    if laplacian_var < 100:
        issues.append("Unscharf/verschwommen")
        recommendations.append("Kamera ruhig halten, schärfer fokussieren")
    
    quality_score = min(100, int(
        (contrast / 100) * 30 +
        (laplacian_var / 500) * 50 +
        (1 - abs(brightness - 127) / 127) * 20
    ))
    
    return {
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "sharpness": round(float(laplacian_var), 2),
        "issues": issues,
        "recommendations": recommendations,
        "quality_score": quality_score
    }


def enhance_image_pil(pil_image: Image.Image, quality_info: Dict) -> Image.Image:
    sharpness_factor = 2.5 if quality_info['sharpness'] < 200 else 1.8
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness_factor)
    
    contrast_factor = 2.0 if quality_info['contrast'] < 50 else 1.5
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)
    
    if quality_info['brightness'] < 100:
        brightness_factor = 1.3
    elif quality_info['brightness'] > 150:
        brightness_factor = 0.8
    else:
        brightness_factor = 1.1
    
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_factor)
    
    return pil_image


def preprocess_ultra_aggressive(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    denoised = cv2.fastNlMeansDenoising(bilateral, h=20)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 3
    )
    
    scaled = cv2.resize(binary, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    
    kernel_close = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    kernel_dilate = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opened, kernel_dilate, iterations=1)
    
    return dilated


def preprocess_aggressive(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    scaled = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def preprocess_standard(image: np.ndarray) -> np.ndarray:
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



def preprocess_inverted(image: np.ndarray) -> np.ndarray:
    processed = preprocess_aggressive(image)
    return cv2.bitwise_not(processed)


def run_easyocr(image: np.ndarray) -> str:
    """Führt EasyOCR aus"""
    if EASYOCR_READER is None:
        return ""
    
    try:
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        results = EASYOCR_READER.readtext(image_rgb, detail=0, paragraph=False)
        text = '\n'.join(results)
        return text
    except Exception as e:
        print(f"⚠ EasyOCR Fehler: {e}")
        return ""


def run_tesseract(image: np.ndarray, psm: int = 6) -> str:
    try:
        config = f'--oem 3 --psm {psm}'
        text = pytesseract.image_to_string(image, lang='deu', config=config)
        return text
    except Exception as e:
        print(f"⚠ Tesseract Fehler: {e}")
        return ""


def run_hybrid_ocr(image: np.ndarray) -> List[Dict]:
    results = []
    
    print("ℹ OCR-Strategie 1: EasyOCR + Ultra-Aggressive")
    img1 = preprocess_ultra_aggressive(image)
    text1 = run_easyocr(img1)
    quality1 = len([c for c in text1 if c.isalpha()])
    results.append({
        "strategy": "easyocr_ultra_aggressive",
        "text": text1,
        "quality": quality1,
        "engine": "easyocr"
    })
    
    if quality1 > 100:
        print(f"✓ Exzellente Qualität erreicht ({quality1} Buchstaben)")
        return results
    
    print("ℹ OCR-Strategie 2: EasyOCR + Aggressive")
    img2 = preprocess_aggressive(image)
    text2 = run_easyocr(img2)
    quality2 = len([c for c in text2 if c.isalpha()])
    results.append({
        "strategy": "easyocr_aggressive",
        "text": text2,
        "quality": quality2,
        "engine": "easyocr"
    })
    
    if quality2 > 100:
        results.sort(key=lambda x: x['quality'], reverse=True)
        return results
    
    print("ℹ OCR-Strategie 3: EasyOCR + Standard")
    img3 = preprocess_standard(image)
    text3 = run_easyocr(img3)
    quality3 = len([c for c in text3 if c.isalpha()])
    results.append({
        "strategy": "easyocr_standard",
        "text": text3,
        "quality": quality3,
        "engine": "easyocr"
    })
    
    best_quality = max(quality1, quality2, quality3)
    if best_quality > 50:
        print("ℹ OCR-Strategie 4: Tesseract + Ultra-Aggressive")
        text4 = run_tesseract(img1, psm=6)
        results.append({
            "strategy": "tesseract_ultra_aggressive_psm6",
            "text": text4,
            "quality": len([c for c in text4 if c.isalpha()]),
            "engine": "tesseract"
        })
    else:
        print("ℹ OCR-Strategie 4: Tesseract + Ultra-Aggressive PSM6")
        text4 = run_tesseract(img1, psm=6)
        results.append({
            "strategy": "tesseract_ultra_aggressive_psm6",
            "text": text4,
            "quality": len([c for c in text4 if c.isalpha()]),
            "engine": "tesseract"
        })
        
        print("ℹ OCR-Strategie 5: Tesseract + Aggressive PSM4")
        text5 = run_tesseract(img2, psm=4)
        results.append({
            "strategy": "tesseract_aggressive_psm4",
            "text": text5,
            "quality": len([c for c in text5 if c.isalpha()]),
            "engine": "tesseract"
        })
        
        print("ℹ OCR-Strategie 6: EasyOCR + Inverted")
        img6 = preprocess_inverted(image)
        text6 = run_easyocr(img6)
        results.append({
            "strategy": "easyocr_inverted",
            "text": text6,
            "quality": len([c for c in text6 if c.isalpha()]),
            "engine": "easyocr"
        })
    
    results.sort(key=lambda x: x['quality'], reverse=True)
    print(f"✓ Beste Strategie: {results[0]['strategy']} ({results[0]['quality']} Buchstaben, Engine: {results[0]['engine']})")
    
    return results


def is_likely_location_keyword(word: str) -> bool:
    if not FILTER_ENABLED:
        return False
    
    for keyword in COMPANY_KEYWORDS:
        if CASE_SENSITIVE:
            if keyword in word:
                return True
        else:
            if keyword.lower() in word.lower():
                return True
    return False


def extract_all_capitalized_words(text: str) -> Set[str]:
    if not text or not text.strip():
        return set()
    
    word_pool = set()
    
    for line in text.split('\n'):
        clean = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', line)
        words = [w.strip() for w in clean.split() if len(w.strip()) >= MIN_WORD_LENGTH]
        
        for word in words:
            if word and word[0].isupper():
                if not is_likely_location_keyword(word):
                    word_pool.add(word)
    
    print(f"ℹ Word Pool: {len(word_pool)} großgeschriebene Wörter gefunden: {sorted(word_pool)}")
    return word_pool


def create_name_combinations(word_pool: Set[str]) -> List[str]:
    words_list = sorted(list(word_pool))
    combinations_list = []
    
    combinations_list.extend(words_list)
    
    if len(words_list) >= 2:
        for combo in combinations(words_list, 2):
            combinations_list.append(f"{combo[0]} {combo[1]}")
    
    if len(words_list) >= 3 and len(words_list) <= 8:
        for combo in combinations(words_list, 3):
            combinations_list.append(f"{combo[0]} {combo[1]} {combo[2]}")
    
    print(f"ℹ {len(combinations_list)} Kombinationen erstellt aus Word Pool")
    return combinations_list


def filter_by_location_match(candidates: List[str]) -> List[str]:
    if not FILTER_ENABLED or not KNOWN_LOCATIONS:
        return candidates
    
    filtered = []
    
    for candidate in candidates:
        match = process.extractOne(
            candidate,
            KNOWN_LOCATIONS,
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] >= LOCATION_FUZZY_THRESHOLD:
            print(f"ℹ Gefiltert (Standort): '{candidate}' → '{match[0]}' ({match[1]}%)")
        else:
            filtered.append(candidate)
    
    return filtered


def match_word_pool_to_recipients(word_pool: Set[str]) -> Dict:
    if not word_pool:
        return {
            "name": "Keine Wörter gefunden",
            "confidence": 0,
            "ocr_text": None,
            "method": "no_words",
            "matched_from_list": False
        }
    
    if not KNOWN_RECIPIENTS:
        return {
            "name": "Keine Empfängerliste verfügbar",
            "confidence": 0,
            "ocr_text": None,
            "method": "no_recipient_list",
            "matched_from_list": False
        }
    
    word_pool_string = " ".join(sorted(word_pool))
    print(f"ℹ Matching Word Pool gegen {len(KNOWN_RECIPIENTS)} Empfänger...")
    print(f"ℹ Word Pool String: '{word_pool_string}'")
    
    scorers = {
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "partial_token_set_ratio": fuzz.partial_token_set_ratio,
        "partial_ratio": fuzz.partial_ratio
    }
    
    best_match = None
    best_score = 0
    best_method = None
    
    for method_name, scorer in scorers.items():
        match = process.extractOne(
            word_pool_string,
            KNOWN_RECIPIENTS,
            scorer=scorer
        )
        
        if match and match[1] > best_score:
            best_score = match[1]
            best_match = match[0]
            best_method = f"word_pool_{method_name}"
    
    combinations_list = create_name_combinations(word_pool)
    combinations_list = filter_by_location_match(combinations_list)
    
    for candidate in combinations_list:
        for method_name, scorer in scorers.items():
            match = process.extractOne(
                candidate,
                KNOWN_RECIPIENTS,
                scorer=scorer
            )
            
            if match and match[1] > best_score:
                best_score = match[1]
                best_match = match[0]
                best_method = f"combination_{method_name}"
                print(f"ℹ Besserer Match: '{candidate}' → '{best_match}' ({best_score}%, {best_method})")
    
    for word in word_pool:
        for recipient in KNOWN_RECIPIENTS:
            recipient_parts = recipient.split()
            
            for part in recipient_parts:
                score = fuzz.ratio(word.lower(), part.lower())
                
                if score > best_score and score >= 85:
                    best_score = score
                    best_match = recipient
                    best_method = f"single_word_match_{word}"
                    print(f"ℹ Einzelwort-Match: '{word}' → '{recipient}' ({score}%)")
    
    if best_match:
        print(f"✓ Bester Match: '{word_pool_string}' → '{best_match}' (Score: {best_score}%, Methode: {best_method})")
    
    if best_match and best_score >= FUZZY_THRESHOLD:
        return {
            "name": best_match,
            "confidence": best_score,
            "ocr_text": word_pool_string,
            "method": best_method,
            "word_pool": sorted(list(word_pool)),
            "matched_from_list": True,
            "fuzzy_threshold_used": FUZZY_THRESHOLD
        }
    elif ENABLE_FALLBACK and best_match and best_score >= 50:
        return {
            "name": best_match,
            "confidence": best_score,
            "ocr_text": word_pool_string,
            "method": f"{best_method}_fallback",
            "word_pool": sorted(list(word_pool)),
            "matched_from_list": True,
            "warning": f"Niedrige Übereinstimmung ({best_score}%). Bitte prüfen.",
            "fuzzy_threshold_used": FUZZY_THRESHOLD
        }
    else:
        return {
            "name": "Keinen passenden Empfänger gefunden",
            "confidence": best_score if best_match else 0,
            "ocr_text": word_pool_string,
            "method": "no_sufficient_match",
            "word_pool": sorted(list(word_pool)),
            "matched_from_list": False,
            "warning": f"Bester Match: '{best_match}' mit {best_score}% (Threshold: {FUZZY_THRESHOLD}%)",
            "recommendation": "Prüfen Sie ob der Empfänger in recipients.json enthalten ist."
        }


def extract_recipient_from_text(text: str) -> Dict:
    word_pool = extract_all_capitalized_words(text)
    
    result = match_word_pool_to_recipients(word_pool)
    
    return result


def extract_recipient_multi_strategy(ocr_results: List[Dict]) -> Dict:
    all_matches = []
    
    for ocr_result in ocr_results:
        text = ocr_result["text"]
        
        result = extract_recipient_from_text(text)
        result["ocr_strategy"] = ocr_result["strategy"]
        result["ocr_quality"] = ocr_result["quality"]
        result["ocr_engine"] = ocr_result["engine"]
        
        all_matches.append(result)
    
    all_matches.sort(
        key=lambda x: (x.get("matched_from_list", False), x.get("confidence", 0)),
        reverse=True
    )
    
    best_result = all_matches[0]
    
    best_result["total_strategies_tested"] = len(ocr_results)
    best_result["strategies_with_match"] = sum(1 for m in all_matches if m.get("matched_from_list", False))
    
    return best_result


@router.post("/upload-image")
async def upload_image(image_data: ImageBase64):
    start_time = time.time()
    
    try:
        if not KNOWN_RECIPIENTS:
            raise HTTPException(
                status_code=500,
                detail="CRITICAL: Keine Empfängerliste geladen."
            )
        
        img_bytes = base64.b64decode(image_data.img_body_base64)
        pil_image = Image.open(BytesIO(img_bytes))
        
        pil_image = convert_heic_to_rgb(pil_image)
        
        numpy_image = np.array(pil_image)

        quality_info = check_image_quality(numpy_image)
        print(f"ℹ Bildqualität: {quality_info['quality_score']}/100")

        pil_image = enhance_image_pil(pil_image, quality_info)
        numpy_image = np.array(pil_image)

        ocr_results = run_hybrid_ocr(numpy_image)

        recipient_result = extract_recipient_multi_strategy(ocr_results)

        best_ocr_text = ocr_results[0]["text"]
        
        processing_time = round(time.time() - start_time, 2)
        print(f"✓ Verarbeitung abgeschlossen in {processing_time}s")

        return {
            "success": True,
            "message": "Bild erfolgreich verarbeitet (Word-Pool-Matching)",
            "processing_time_seconds": processing_time,
            "image_size_bytes": len(img_bytes),
            "image_format": pil_image.format or "Unknown",
            "quality_info": quality_info,
            "ocr_text": best_ocr_text,
            "ocr_strategies_tested": len(ocr_results),
            "best_ocr_strategy": ocr_results[0]["strategy"],
            "best_ocr_engine": ocr_results[0]["engine"],
            "recipient": recipient_result["name"],
            "recipient_details": recipient_result,
            "image_base64": image_data.img_body_base64,
            "available_recipients_count": len(KNOWN_RECIPIENTS)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ Fehler: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__
            }
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
        "recipients_count": len(KNOWN_RECIPIENTS)
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
        "locations_count": len(KNOWN_LOCATIONS)
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
        "easyocr_available": EASYOCR_READER is not None,
        "tesseract_available": True,
        "recipients_loaded": len(KNOWN_RECIPIENTS) > 0,
        "locations_loaded": len(KNOWN_LOCATIONS) > 0,
        "recipients_count": len(KNOWN_RECIPIENTS),
        "locations_count": len(KNOWN_LOCATIONS),
        "filter_enabled": FILTER_ENABLED,
        "fuzzy_threshold": FUZZY_THRESHOLD,
        "version": "3.0.0-word-pool-matching"
    }


@router.get("/stats")
async def get_statistics():
    return {
        "configuration": {
            "fuzzy_threshold": FUZZY_THRESHOLD,
            "min_word_length": MIN_WORD_LENGTH,
            "enable_fallback": ENABLE_FALLBACK,
            "location_fuzzy_threshold": LOCATION_FUZZY_THRESHOLD,
            "filter_enabled": FILTER_ENABLED,
            "case_sensitive": CASE_SENSITIVE
        },
        "data": {
            "recipients_count": len(KNOWN_RECIPIENTS),
            "locations_count": len(KNOWN_LOCATIONS),
            "keywords_count": len(COMPANY_KEYWORDS)
        },
        "ocr_engines": {
            "easyocr": EASYOCR_READER is not None,
            "tesseract": True
        },
        "features": {
            "word_pool_matching": True,
            "separated_name_handling": True
        }
    }