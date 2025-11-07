from fastapi import APIRouter, HTTPException
from app.routers.ocr.model import ImageBase64
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from pathlib import Path
import time
from typing import List, Dict

from pillow_heif import register_heif_opener

from .preprocessing import (
    optimize_image_size,
    check_image_quality,
    enhance_image_adaptive,
    preprocess_ultra_aggressive,
    preprocess_aggressive,
    preprocess_standard,
    preprocess_inverted
)

from .ocr_engines import run_hybrid_ocr_parallel

from .matching import (
    extract_capitalized_words,
    match_word_pool_comprehensive
)

from .utils import (
    convert_heic_to_rgb,
    load_recipients,
    load_locations,
    build_recipient_caches
)

register_heif_opener()

router = APIRouter(
    prefix="/ocr",
    tags=["ocr"]
)

DATA_DIR = Path(__file__).parent.parent / "data"
RECIPIENTS_FILE = DATA_DIR / "recipients.json"
LOCATIONS_FILE = DATA_DIR / "locations.json"

KNOWN_RECIPIENTS, RECIPIENT_SETTINGS = load_recipients(RECIPIENTS_FILE)
KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS = load_locations(LOCATIONS_FILE)
RECIPIENT_CACHE, NAME_PARTS_CACHE = build_recipient_caches(KNOWN_RECIPIENTS)

FUZZY_THRESHOLD = RECIPIENT_SETTINGS.get("fuzzy_threshold", 65)
MIN_WORD_LENGTH = RECIPIENT_SETTINGS.get("min_word_length", 2)
ENABLE_FALLBACK = RECIPIENT_SETTINGS.get("enable_fallback", True)
LOCATION_FUZZY_THRESHOLD = LOCATION_SETTINGS.get("location_fuzzy_threshold", 85)
FILTER_ENABLED = LOCATION_SETTINGS.get("filter_enabled", True)
CASE_SENSITIVE = LOCATION_SETTINGS.get("case_sensitive", False)

# Global variable for batch progress tracking
BATCH_PROGRESS = {
    "processing": False,
    "current": 0,
    "total": 0,
    "percentage": 0
}


def extract_recipient_from_ocr(text: str) -> Dict:
    word_pool = extract_capitalized_words(
        text,
        MIN_WORD_LENGTH,
        COMPANY_KEYWORDS,
        CASE_SENSITIVE,
        FILTER_ENABLED
    )
    
    result = match_word_pool_comprehensive(
        word_pool,
        KNOWN_RECIPIENTS,
        KNOWN_LOCATIONS,
        COMPANY_KEYWORDS,
        LOCATION_FUZZY_THRESHOLD,
        FILTER_ENABLED,
        FUZZY_THRESHOLD,
        ENABLE_FALLBACK,
        NAME_PARTS_CACHE
    )
    
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
        
        preprocess_funcs = {
            'ultra_aggressive': preprocess_ultra_aggressive,
            'aggressive': preprocess_aggressive,
            'standard': preprocess_standard,
            'inverted': preprocess_inverted
        }
        
        ocr_results = run_hybrid_ocr_parallel(numpy_image, quality_info['preprocessing_mode'], preprocess_funcs)
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
    global RECIPIENT_CACHE, NAME_PARTS_CACHE
    
    KNOWN_RECIPIENTS, RECIPIENT_SETTINGS = load_recipients(RECIPIENTS_FILE)
    RECIPIENT_CACHE, NAME_PARTS_CACHE = build_recipient_caches(KNOWN_RECIPIENTS)
    FUZZY_THRESHOLD = RECIPIENT_SETTINGS.get("fuzzy_threshold", 65)
    MIN_WORD_LENGTH = RECIPIENT_SETTINGS.get("min_word_length", 2)
    ENABLE_FALLBACK = RECIPIENT_SETTINGS.get("enable_fallback", True)
    
    return {"success": True, "message": "Recipients reloaded", "recipients_count": len(KNOWN_RECIPIENTS)}


@router.post("/reload-locations")
async def reload_locations():
    global KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS
    global LOCATION_FUZZY_THRESHOLD, FILTER_ENABLED, CASE_SENSITIVE
    
    KNOWN_LOCATIONS, COMPANY_KEYWORDS, LOCATION_SETTINGS = load_locations(LOCATIONS_FILE)
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
    from .ocr_engines import EASYOCR_READER
    
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
        "version": "4.1.0-batch-progress"
    }


@router.get("/batch-progress")
async def get_batch_progress():
    """
    Gibt den aktuellen Fortschritt der Batch-Verarbeitung zurück
    """
    return {
        "processing": BATCH_PROGRESS["processing"],
        "current": BATCH_PROGRESS["current"],
        "total": BATCH_PROGRESS["total"],
        "percentage": BATCH_PROGRESS["percentage"]
    }


@router.post("/convert-heic")
async def convert_heic(image_data: ImageBase64):
    """
    Konvertiert ein HEIC-Bild zu JPG für die Browser-Vorschau
    """
    try:
        img_bytes = base64.b64decode(image_data.img_body_base64)
        pil_image = Image.open(BytesIO(img_bytes))
        
        # HEIC zu RGB konvertieren
        pil_image = convert_heic_to_rgb(pil_image)
        
        # Als JPG zurückgeben
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        jpg_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "success": True,
            "converted_base64": jpg_base64,
            "format": "JPEG",
            "size": pil_image.size
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"HEIC conversion error: {error_trace}")
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})


@router.post("/upload-images-batch")
async def upload_images_batch(images_data: List[ImageBase64]):
    """
    Verarbeitet mehrere Bilder gleichzeitig mit Progress-Tracking
    """
    global BATCH_PROGRESS
    
    results = []
    start_time = time.time()
    
    try:
        if not KNOWN_RECIPIENTS:
            raise HTTPException(status_code=500, detail="No recipient list loaded")
        
        # Initialize progress
        BATCH_PROGRESS["processing"] = True
        BATCH_PROGRESS["current"] = 0
        BATCH_PROGRESS["total"] = len(images_data)
        BATCH_PROGRESS["percentage"] = 0
        
        for idx, image_data in enumerate(images_data):
            try:
                # Verarbeite jedes Bild einzeln
                img_bytes = base64.b64decode(image_data.img_body_base64)
                pil_image = Image.open(BytesIO(img_bytes))
                pil_image = convert_heic_to_rgb(pil_image)
                pil_image = optimize_image_size(pil_image)
                numpy_image = np.array(pil_image)
                
                quality_info = check_image_quality(numpy_image)
                print(f"Image {idx+1}/{len(images_data)}: Quality {quality_info['quality_score']}/100")
                
                pil_image = enhance_image_adaptive(pil_image, quality_info)
                numpy_image = np.array(pil_image)
                
                preprocess_funcs = {
                    'ultra_aggressive': preprocess_ultra_aggressive,
                    'aggressive': preprocess_aggressive,
                    'standard': preprocess_standard,
                    'inverted': preprocess_inverted
                }
                
                ocr_results = run_hybrid_ocr_parallel(numpy_image, quality_info['preprocessing_mode'], preprocess_funcs)
                recipient_result = extract_best_recipient(ocr_results)
                
                results.append({
                    "image_index": idx,
                    "success": True,
                    "recipient": recipient_result["name"],
                    "confidence": recipient_result.get("confidence", 0),
                    "matched_from_list": recipient_result.get("matched_from_list", False),
                    "method": recipient_result.get("method", "unknown"),
                    "ocr_text": recipient_result.get("ocr_text", ""),
                    "word_pool": recipient_result.get("word_pool", []),
                    "quality_score": quality_info['quality_score'],
                    "image_base64": image_data.img_body_base64,
                    "warning": recipient_result.get("warning", None)
                })
                
                # Update progress
                BATCH_PROGRESS["current"] = idx + 1
                BATCH_PROGRESS["percentage"] = int((idx + 1) / len(images_data) * 100)
                
            except Exception as e:
                results.append({
                    "image_index": idx,
                    "success": False,
                    "error": str(e),
                    "image_base64": image_data.img_body_base64
                })
                print(f"Error processing image {idx+1}: {e}")
                
                # Update progress even on error
                BATCH_PROGRESS["current"] = idx + 1
                BATCH_PROGRESS["percentage"] = int((idx + 1) / len(images_data) * 100)
        
        total_time = round(time.time() - start_time, 2)
        successful = sum(1 for r in results if r.get("success", False))
        
        # Reset progress
        BATCH_PROGRESS["processing"] = False
        BATCH_PROGRESS["current"] = 0
        BATCH_PROGRESS["total"] = 0
        BATCH_PROGRESS["percentage"] = 0
        
        return {
            "success": True,
            "message": f"Processed {len(images_data)} images",
            "total_processing_time_seconds": total_time,
            "images_processed": len(images_data),
            "successful_matches": successful,
            "failed_matches": len(images_data) - successful,
            "results": results
        }
        
    except HTTPException:
        BATCH_PROGRESS["processing"] = False
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Batch processing error: {error_trace}")
        BATCH_PROGRESS["processing"] = False
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})


@router.get("/stats")
async def get_statistics():
    from .ocr_engines import EASYOCR_READER
    
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
            "smart_combinations": True,
            "modular_architecture": True,
            "batch_progress_tracking": True
        }
    }