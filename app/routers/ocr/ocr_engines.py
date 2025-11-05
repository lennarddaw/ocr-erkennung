import cv2
import numpy as np
import pytesseract
import easyocr
from typing import List, Dict


print("Initializing EasyOCR...")
try:
    EASYOCR_READER = easyocr.Reader(['de', 'en'], gpu=False, verbose=False)
    print("EasyOCR initialized")
except Exception as e:
    print(f"EasyOCR error: {e}")
    EASYOCR_READER = None


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


def run_hybrid_ocr_parallel(image: np.ndarray, preprocessing_mode: str, preprocess_funcs: Dict) -> List[Dict]:
    results = []
    
    if preprocessing_mode == "ultra_aggressive":
        img1 = preprocess_funcs['ultra_aggressive'](image)
        img2 = preprocess_funcs['aggressive'](image)
    elif preprocessing_mode == "aggressive":
        img1 = preprocess_funcs['aggressive'](image)
        img2 = preprocess_funcs['standard'](image)
    else:
        img1 = preprocess_funcs['standard'](image)
        img2 = preprocess_funcs['aggressive'](image)
    
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
            img_inv = preprocess_funcs['inverted'](image)
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