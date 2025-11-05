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
import hashlib
from itertools import combinations, permutations
from concurrent.futures import ThreadPoolExecutor
import Levenshtein

register_heif_opener()

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

DATA_DIR = Path(__file__).parent.parent / "data"
RECIPIENTS_FILE = DATA_DIR / "recipients.json"
LOCATIONS_FILE = DATA_DIR / "locations.json"

print("Initializing EasyOCR...")
try:
    EASYOCR_READER = easyocr.Reader(['de', 'en'], gpu=False, verbose=False)
    print("EasyOCR initialized")
except Exception as e:
    print(f"EasyOCR error: {e}")
    EASYOCR_READER = None

RECIPIENT_CACHE = {}
NAME_PARTS_CACHE = {}


def load_recipients() -> Tuple[List[str], Dict]:
    try:
        with open(RECIPIENTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            recipients = data.get("recipients", [])
            settings = data.get("settings", {})
            
            global RECIPIENT_CACHE, NAME_PARTS_CACHE
            RECIPIENT_CACHE = {r.lower(): r for r in recipients}
            
            for recipient in recipients:
                parts = recipient.split()
                for part in parts:
                    if len(part) >= 2:
                        part_lower = part.lower()
                        if part_lower not in NAME_PARTS_CACHE:
                            NAME_PARTS_CACHE[part_lower] = []
                        NAME_PARTS_CACHE[part_lower].append(recipient)
            
            print(f"Loaded {len(recipients)} recipients")
            return recipients, settings
    except Exception as e:
        print(f"Error loading recipients: {e}")
        return [], {}


def load_locations() -> Tuple[List[str], List[str], Dict]:
    try:
        with open(LOCATIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            locations = data.get("locations", [])
            keywords = data.get("company_keywords", [])
            settings = data.get("settings", {})
            print(f"Loaded {len(locations)} locations, {len(keywords)} keywords")
            return locations, keywords, settings
    except Exception as e:
        print(f"Error loading locations: {e}")
        return [], [], {"filter_enabled": False}


KNOWN_RECIPIENTS, RECIPIENT_SETTINGS = load_recipients()
KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS = load_locations()

FUZZY_THRESHOLD = RECIPIENT_SETTINGS.get("fuzzy_threshold", 65)
MIN_WORD_LENGTH = RECIPIENT_SETTINGS.get("min_word_length", 2)
ENABLE_FALLBACK = RECIPIENT_SETTINGS.get("enable_fallback", True)
LOCATION_FUZZY_THRESHOLD = LOCATION_SETTINGS.get("location_fuzzy_threshold", 85)
FILTER_ENABLED = LOCATION_SETTINGS.get("filter_enabled", True)
CASE_SENSITIVE = LOCATION_SETTINGS.get("case_sensitive", False)

TITLE_PREFIXES = {'dr', 'prof', 'mr', 'mrs', 'ms', 'dr.', 'prof.'}
EXECUTOR = ThreadPoolExecutor(max_workers=3)


def convert_heic_to_rgb(pil_image: Image.Image) -> Image.Image:
    try:
        if pil_image.format in ['HEIF', 'HEIC']:
            pil_image = pil_image.convert('RGB')
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        return pil_image.convert('RGB')


def optimize_image_size(pil_image: Image.Image) -> Image.Image:
    max_dimension = 2000
    width, height = pil_image.size
    
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        new_size = (int(width * ratio), int(height * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    return pil_image


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
        issues.append("Too dark")
        recommendations.append("Better lighting needed")
    elif brightness > 200:
        issues.append("Overexposed")
        recommendations.append("Reduce exposure")
    
    if contrast < 30:
        issues.append("Low contrast")
        recommendations.append("Increase text/background contrast")
    
    if laplacian_var < 100:
        issues.append("Blurry")
        recommendations.append("Focus camera, hold steady")
    
    quality_score = min(100, int(
        (contrast / 100) * 30 +
        (laplacian_var / 500) * 50 +
        (1 - abs(brightness - 127) / 127) * 20
    ))
    
    preprocessing_mode = "ultra_aggressive" if quality_score < 40 else "aggressive" if quality_score < 70 else "standard"
    
    return {
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "sharpness": round(float(laplacian_var), 2),
        "issues": issues,
        "recommendations": recommendations,
        "quality_score": quality_score,
        "preprocessing_mode": preprocessing_mode
    }


def enhance_image_adaptive(pil_image: Image.Image, quality_info: Dict) -> Image.Image:
    mode = quality_info.get("preprocessing_mode", "standard")
    
    if mode == "ultra_aggressive":
        sharpness_factor = 3.0
        contrast_factor = 2.5
        brightness_adjust = 1.4 if quality_info['brightness'] < 100 else 0.7
    elif mode == "aggressive":
        sharpness_factor = 2.5
        contrast_factor = 2.0
        brightness_adjust = 1.3 if quality_info['brightness'] < 100 else 0.8
    else:
        sharpness_factor = 1.8
        contrast_factor = 1.5
        brightness_adjust = 1.1 if quality_info['brightness'] < 100 else 0.9
    
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness_factor)
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast_factor)
    
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness_adjust)
    
    return pil_image


def preprocess_ultra_aggressive(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    denoised = cv2.fastNlMeansDenoising(bilateral, h=25)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        17, 4
    )
    
    scaled = cv2.resize(binary, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    
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
    
    denoised = cv2.fastNlMeansDenoising(gray, h=18)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        13, 3
    )
    
    scaled = cv2.resize(binary, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def preprocess_standard(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    denoised = cv2.fastNlMeansDenoising(gray, h=12)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scaled = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def preprocess_inverted(image: np.ndarray) -> np.ndarray:
    processed = preprocess_aggressive(image)
    return cv2.bitwise_not(processed)


def run_easyocr(image: np.ndarray) -> str:
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
        return ""


def run_tesseract(image: np.ndarray, psm: int = 6) -> str:
    try:
        config = f'--oem 3 --psm {psm}'
        text = pytesseract.image_to_string(image, lang='deu', config=config)
        return text
    except Exception as e:
        return ""


def run_hybrid_ocr_parallel(image: np.ndarray, preprocessing_mode: str) -> List[Dict]:
    results = []
    
    if preprocessing_mode == "ultra_aggressive":
        img1 = preprocess_ultra_aggressive(image)
        img2 = preprocess_aggressive(image)
    elif preprocessing_mode == "aggressive":
        img1 = preprocess_aggressive(image)
        img2 = preprocess_standard(image)
    else:
        img1 = preprocess_standard(image)
        img2 = preprocess_aggressive(image)
    
    print(f"OCR Strategy 1: EasyOCR + {preprocessing_mode}")
    text1 = run_easyocr(img1)
    quality1 = len([c for c in text1 if c.isalpha()])
    results.append({
        "strategy": f"easyocr_{preprocessing_mode}",
        "text": text1,
        "quality": quality1,
        "engine": "easyocr"
    })
    
    if quality1 > 120:
        print(f"Excellent quality ({quality1} chars), stopping")
        return results
    
    print(f"OCR Strategy 2: EasyOCR + fallback")
    text2 = run_easyocr(img2)
    quality2 = len([c for c in text2 if c.isalpha()])
    results.append({
        "strategy": f"easyocr_fallback",
        "text": text2,
        "quality": quality2,
        "engine": "easyocr"
    })
    
    if quality2 > 120:
        results.sort(key=lambda x: x['quality'], reverse=True)
        return results
    
    best_quality = max(quality1, quality2)
    
    if best_quality > 60:
        print("OCR Strategy 3: Tesseract backup")
        text3 = run_tesseract(img1, psm=6)
        results.append({
            "strategy": f"tesseract_{preprocessing_mode}",
            "text": text3,
            "quality": len([c for c in text3 if c.isalpha()]),
            "engine": "tesseract"
        })
    else:
        print("OCR Strategy 3: Tesseract PSM6")
        text3 = run_tesseract(img1, psm=6)
        results.append({
            "strategy": f"tesseract_{preprocessing_mode}_psm6",
            "text": text3,
            "quality": len([c for c in text3 if c.isalpha()]),
            "engine": "tesseract"
        })
        
        print("OCR Strategy 4: Tesseract PSM4")
        text4 = run_tesseract(img2, psm=4)
        results.append({
            "strategy": f"tesseract_fallback_psm4",
            "text": text4,
            "quality": len([c for c in text4 if c.isalpha()]),
            "engine": "tesseract"
        })
        
        if best_quality < 40:
            print("OCR Strategy 5: Inverted")
            img_inv = preprocess_inverted(image)
            text5 = run_easyocr(img_inv)
            results.append({
                "strategy": "easyocr_inverted",
                "text": text5,
                "quality": len([c for c in text5 if c.isalpha()]),
                "engine": "easyocr"
            })
    
    results.sort(key=lambda x: x['quality'], reverse=True)
    print(f"Best: {results[0]['strategy']} ({results[0]['quality']} chars, {results[0]['engine']})")
    
    return results


def normalize_word(word: str) -> str:
    word_lower = word.lower()
    if word_lower in TITLE_PREFIXES:
        return None
    return word


def is_location_keyword(word: str) -> bool:
    if not FILTER_ENABLED:
        return False
    
    word_lower = word.lower()
    for keyword in COMPANY_KEYWORDS:
        if CASE_SENSITIVE:
            if keyword in word:
                return True
        else:
            if keyword.lower() in word_lower:
                return True
    return False


def extract_capitalized_words(text: str) -> Set[str]:
    if not text or not text.strip():
        return set()
    
    word_pool = set()
    
    for line in text.split('\n'):
        clean = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', line)
        words = [w.strip() for w in clean.split() if len(w.strip()) >= MIN_WORD_LENGTH]
        
        for word in words:
            if word and word[0].isupper():
                normalized = normalize_word(word)
                if normalized and not is_location_keyword(word):
                    word_pool.add(word)
    
    print(f"Word pool: {len(word_pool)} words: {sorted(word_pool)}")
    return word_pool


def create_smart_combinations(word_pool: Set[str]) -> List[Tuple[str, int]]:
    words_list = sorted(list(word_pool))
    combinations_with_priority = []
    
    for word in words_list:
        combinations_with_priority.append((word, 1))
    
    if len(words_list) >= 2:
        for combo in combinations(words_list, 2):
            combinations_with_priority.append((f"{combo[0]} {combo[1]}", 3))
        
        for perm in permutations(words_list, 2):
            combinations_with_priority.append((f"{perm[0]} {perm[1]}", 2))
    
    if len(words_list) >= 3 and len(words_list) <= 6:
        for combo in combinations(words_list, 3):
            combinations_with_priority.append((f"{combo[0]} {combo[1]} {combo[2]}", 4))
    
    combinations_with_priority.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Created {len(combinations_with_priority)} combinations")
    return combinations_with_priority


def filter_by_locations(candidates: List[str]) -> List[str]:
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
            print(f"Filtered location: '{candidate}' -> '{match[0]}' ({match[1]}%)")
        else:
            filtered.append(candidate)
    
    return filtered


def advanced_fuzzy_match(candidate: str, recipients: List[str]) -> Tuple[Optional[str], int, str]:
    scorers = {
        "token_set_ratio": (fuzz.token_set_ratio, 1.0),
        "token_sort_ratio": (fuzz.token_sort_ratio, 0.95),
        "partial_token_set_ratio": (fuzz.partial_token_set_ratio, 0.9),
        "partial_ratio": (fuzz.partial_ratio, 0.85),
        "ratio": (fuzz.ratio, 0.8)
    }
    
    best_match = None
    best_score = 0
    best_method = None
    
    for method_name, (scorer, weight) in scorers.items():
        match = process.extractOne(candidate, recipients, scorer=scorer)
        if match:
            weighted_score = match[1] * weight
            if weighted_score > best_score:
                best_score = int(match[1])
                best_match = match[0]
                best_method = method_name
    
    return best_match, best_score, best_method


def levenshtein_name_parts_match(word_pool: Set[str], recipients: List[str]) -> Tuple[Optional[str], int, str]:
    best_recipient = None
    best_score = 0
    
    for recipient in recipients:
        recipient_parts = set(part.lower() for part in recipient.split())
        word_pool_lower = set(w.lower() for w in word_pool)
        
        common_parts = recipient_parts & word_pool_lower
        
        if common_parts:
            similarity = len(common_parts) / len(recipient_parts)
            score = int(similarity * 100)
            
            if score > best_score:
                best_score = score
                best_recipient = recipient
    
    if best_score >= 50:
        return best_recipient, best_score, "name_parts_match"
    
    for word in word_pool:
        word_lower = word.lower()
        if word_lower in NAME_PARTS_CACHE:
            possible_recipients = NAME_PARTS_CACHE[word_lower]
            for recipient in possible_recipients:
                for part in recipient.split():
                    distance = Levenshtein.distance(word_lower, part.lower())
                    max_len = max(len(word_lower), len(part))
                    similarity = (max_len - distance) / max_len
                    score = int(similarity * 100)
                    
                    if score > best_score and score >= 80:
                        best_score = score
                        best_recipient = recipient
    
    if best_recipient:
        return best_recipient, best_score, "levenshtein_parts"
    
    return None, 0, "none"


def match_word_pool_comprehensive(word_pool: Set[str]) -> Dict:
    if not word_pool:
        return {
            "name": "No words found",
            "confidence": 0,
            "ocr_text": None,
            "method": "no_words",
            "matched_from_list": False
        }
    
    if not KNOWN_RECIPIENTS:
        return {
            "name": "No recipient list available",
            "confidence": 0,
            "ocr_text": None,
            "method": "no_recipient_list",
            "matched_from_list": False
        }
    
    word_pool_string = " ".join(sorted(word_pool))
    print(f"Matching word pool: '{word_pool_string}'")
    
    best_match = None
    best_score = 0
    best_method = None
    
    match, score, method = advanced_fuzzy_match(word_pool_string, KNOWN_RECIPIENTS)
    if score > best_score:
        best_match = match
        best_score = score
        best_method = f"word_pool_{method}"
    
    combinations = create_smart_combinations(word_pool)
    filtered_combinations = filter_by_locations([c[0] for c in combinations])
    
    for candidate in filtered_combinations[:20]:
        match, score, method = advanced_fuzzy_match(candidate, KNOWN_RECIPIENTS)
        if score > best_score:
            best_match = match
            best_score = score
            best_method = f"combination_{method}"
            print(f"Better: '{candidate}' -> '{best_match}' ({best_score}%, {best_method})")
    
    parts_match, parts_score, parts_method = levenshtein_name_parts_match(word_pool, KNOWN_RECIPIENTS)
    if parts_score > best_score:
        best_match = parts_match
        best_score = parts_score
        best_method = parts_method
        print(f"Name parts: '{word_pool_string}' -> '{best_match}' ({best_score}%)")
    
    if best_match:
        print(f"Best: '{word_pool_string}' -> '{best_match}' ({best_score}%, {best_method})")
    
    adjusted_threshold = FUZZY_THRESHOLD - 5 if best_method and "levenshtein" in best_method else FUZZY_THRESHOLD
    
    if best_match and best_score >= adjusted_threshold:
        return {
            "name": best_match,
            "confidence": best_score,
            "ocr_text": word_pool_string,
            "method": best_method,
            "word_pool": sorted(list(word_pool)),
            "matched_from_list": True,
            "fuzzy_threshold_used": adjusted_threshold
        }
    elif ENABLE_FALLBACK and best_match and best_score >= 45:
        return {
            "name": best_match,
            "confidence": best_score,
            "ocr_text": word_pool_string,
            "method": f"{best_method}_fallback",
            "word_pool": sorted(list(word_pool)),
            "matched_from_list": True,
            "warning": f"Low confidence ({best_score}%). Please verify.",
            "fuzzy_threshold_used": adjusted_threshold
        }
    else:
        return {
            "name": "No suitable recipient found",
            "confidence": best_score if best_match else 0,
            "ocr_text": word_pool_string,
            "method": "no_sufficient_match",
            "word_pool": sorted(list(word_pool)),
            "matched_from_list": False,
            "warning": f"Best: '{best_match}' with {best_score}% (Threshold: {adjusted_threshold}%)",
            "recommendation": "Check if recipient is in recipients.json"
        }


def extract_recipient_from_ocr(text: str) -> Dict:
    word_pool = extract_capitalized_words(text)
    result = match_word_pool_comprehensive(word_pool)
    return result


def extract_best_recipient(ocr_results: List[Dict]) -> Dict:
    all_matches = []
    
    for ocr_result in ocr_results:
        text = ocr_result["text"]
        result = extract_recipient_from_ocr(text)
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
            raise HTTPException(status_code=500, detail="No recipient list loaded")
        
        img_bytes = base64.b64decode(image_data.img_body_base64)
        pil_image = Image.open(BytesIO(img_bytes))
        pil_image = convert_heic_to_rgb(pil_image)
        pil_image = optimize_image_size(pil_image)
        numpy_image = np.array(pil_image)
        
        quality_info = check_image_quality(numpy_image)
        print(f"Quality: {quality_info['quality_score']}/100, mode: {quality_info['preprocessing_mode']}")
        
        pil_image = enhance_image_adaptive(pil_image, quality_info)
        numpy_image = np.array(pil_image)
        
        ocr_results = run_hybrid_ocr_parallel(numpy_image, quality_info['preprocessing_mode'])
        recipient_result = extract_best_recipient(ocr_results)
        
        best_ocr_text = ocr_results[0]["text"]
        processing_time = round(time.time() - start_time, 2)
        
        print(f"Processing complete in {processing_time}s")
        
        return {
            "success": True,
            "message": "Image processed with word-pool matching",
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
        print(f"Error: {error_trace}")
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})


@router.get("/recipients")
async def get_recipients():
    return {"recipients": KNOWN_RECIPIENTS, "count": len(KNOWN_RECIPIENTS), "settings": RECIPIENT_SETTINGS}


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
    FUZZY_THRESHOLD = RECIPIENT_SETTINGS.get("fuzzy_threshold", 65)
    MIN_WORD_LENGTH = RECIPIENT_SETTINGS.get("min_word_length", 2)
    ENABLE_FALLBACK = RECIPIENT_SETTINGS.get("enable_fallback", True)
    
    return {"success": True, "message": "Recipients reloaded", "recipients_count": len(KNOWN_RECIPIENTS)}


@router.post("/reload-locations")
async def reload_locations():
    global KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS
    global LOCATION_FUZZY_THRESHOLD, FILTER_ENABLED, CASE_SENSITIVE
    
    KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS = load_locations()
    LOCATION_FUZZY_THRESHOLD = LOCATION_SETTINGS.get("location_fuzzy_threshold", 85)
    FILTER_ENABLED = LOCATION_SETTINGS.get("filter_enabled", True)
    CASE_SENSITIVE = LOCATION_SETTINGS.get("case_sensitive", False)
    
    return {"success": True, "message": "Locations reloaded", "locations_count": len(KNOWN_LOCATIONS)}


@router.post("/reload-all")
async def reload_all():
    recipients_result = await reload_recipients()
    locations_result = await reload_locations()
    return {"success": True, "message": "All lists reloaded", "recipients": recipients_result, "locations": locations_result}


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
        "cache_size": len(RECIPIENT_CACHE),
        "version": "4.0.0-optimized"
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
            "keywords_count": len(COMPANY_KEYWORDS),
            "cached_recipients": len(RECIPIENT_CACHE),
            "cached_name_parts": len(NAME_PARTS_CACHE)
        },
        "ocr_engines": {"easyocr": EASYOCR_READER is not None, "tesseract": True},
        "features": {
            "word_pool_matching": True,
            "levenshtein_matching": True,
            "name_parts_cache": True,
            "adaptive_preprocessing": True,
            "parallel_processing": True,
            "smart_combinations": True,
            "permutations": True
        }
    }