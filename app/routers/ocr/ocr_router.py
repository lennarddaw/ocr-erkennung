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

def preprocess_image_v1(image):
    """Methode 1: Otsu mit CLAHE"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height = gray.shape[0]
    gray = gray[0:int(height*0.4), :]
    
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    scaled = cv2.resize(binary, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    return scaled


def preprocess_image_v2(image):
    """Methode 2: Adaptive Threshold"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height = gray.shape[0]
    gray = gray[0:int(height*0.35), :]
    
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    scaled = cv2.resize(adaptive, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return scaled


def is_likely_german_address(text: str) -> bool:
    """Prüft ob Text wie eine deutsche Adresse aussieht"""
    if not text:
        return False
    
    address_patterns = [
        r'\bStr\b',
        r'\bStraße\b',
        r'\bstrasse\b',
        r'\bAllee\b',
        r'\bWeg\b',
        r'\bPlatz\b',
        r'\bGasse\b',
        r'\d{5}',  # PLZ
        r'\b\d{1,4}\s*[a-zA-Z]?\b.*(?:Str|straße)',  # Hausnummer + Straße
    ]
    
    matches = 0
    for pattern in address_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1
    
    return matches >= 1


def calculate_name_score(w1: str, w2: str, line_idx: int, lines: list, full_text: str) -> int:
    """
    Berechnet Score für einen Namen-Kandidaten
    Höherer Score = wahrscheinlicher ein echter Name
    """
    score = 0
    name = f"{w1} {w2}"
    
    # 1. Position: Frühere Zeilen bevorzugen (aber nicht zu stark)
    score += max(0, 15 - line_idx * 2)
    
    # 2. Längen-Balance: Ähnliche Längen bevorzugen
    len_diff = abs(len(w1) - len(w2))
    if len_diff <= 3:
        score += 10
    elif len_diff > 8:
        score -= 5
    
    # 3. Typische deutsche Nachnamen-Endungen
    name_endings = ['er', 'mann', 'ner', 'berg', 'stein', 'feld', 'wald', 'schwarz']
    if any(w2.lower().endswith(end) for end in name_endings):
        score += 15
    
    # 4. Häufigkeit: Name sollte nicht zu oft vorkommen
    count = full_text.lower().count(name.lower())
    if count == 1:
        score += 5
    elif count > 2:
        score -= 10
    
    # 5. Nächste Zeile ist Adresse? SEHR wichtig!
    if line_idx + 1 < len(lines):
        next_line = lines[line_idx + 1]
        if is_likely_german_address(next_line):
            score += 40  # Starker Indikator!
    
    # 6. Gleiche Zeile hat Adress-Elemente? Dann wahrscheinlich KEIN Name
    current_line = lines[line_idx] if line_idx < len(lines) else ""
    if is_likely_german_address(current_line):
        score -= 20
    
    # 7. Großbuchstaben-Ratio: Nur erster Buchstabe sollte groß sein
    upper_count_w1 = sum(1 for c in w1 if c.isupper())
    upper_count_w2 = sum(1 for c in w2 if c.isupper())
    
    if upper_count_w1 == 1 and upper_count_w2 == 1:
        score += 10
    elif upper_count_w1 > 2 or upper_count_w2 > 2:
        score -= 20  # Zu viele Großbuchstaben = wahrscheinlich Müll
    
    # 8. Wortlänge: Sehr kurze oder sehr lange Wörter sind verdächtig
    if 4 <= len(w1) <= 12 and 4 <= len(w2) <= 12:
        score += 5
    
    # 9. Vokal-Check: Deutsche Namen haben normalerweise Vokale
    vowels = 'aeiouäöü'
    has_vowels_w1 = any(c.lower() in vowels for c in w1)
    has_vowels_w2 = any(c.lower() in vowels for c in w2)
    
    if has_vowels_w1 and has_vowels_w2:
        score += 10
    else:
        score -= 15
    
    # 10. Bekannte deutsche Vornamen (klein halten!)
    common_first_names = ['michael', 'thomas', 'andreas', 'peter', 'wolfgang', 
                          'klaus', 'jürgen', 'christian', 'frank', 'stefan']
    if w1.lower() in common_first_names:
        score += 20
    
    return score


def extract_recipient_with_scoring(text: str) -> dict:
    """
    Findet Name mit Scoring-System
    Gibt auch Debug-Info zurück
    """
    if not text or not text.strip():
        return {"name": "Kein Text erkannt", "score": 0, "candidates": []}
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    candidates = []
    
    # Sammle alle Kandidaten
    for line_idx, line in enumerate(lines):
        clean = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', line)
        words = clean.split()
        
        for i in range(len(words) - 1):
            w1, w2 = words[i].strip(), words[i + 1].strip()
            
            # Basis-Validierung
            if not (len(w1) >= 3 and len(w2) >= 3 and
                    3 <= len(w1) <= 15 and 3 <= len(w2) <= 15 and
                    w1[0].isupper() and w2[0].isupper() and
                    w1.isalpha() and w2.isalpha()):
                continue
            
            name = f"{w1} {w2}"
            score = calculate_name_score(w1, w2, line_idx, lines, text)
            
            candidates.append({
                "name": name,
                "score": score,
                "line": line_idx
            })
    
    # Sortiere nach Score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Bester Kandidat mit positivem Score
    if candidates and candidates[0]['score'] > 0:
        return {
            "name": candidates[0]['name'],
            "score": candidates[0]['score'],
            "candidates": candidates[:5]  # Top 5 für Debug
        }
    
    return {
        "name": "Keinen Empfänger gefunden",
        "score": 0,
        "candidates": candidates[:5] if candidates else []
    }


@router.post("/upload-image")
async def upload_image(image_data: ImageBase64):
    """OCR mit Multi-Method Approach"""
    try:
        img_bytes = base64.b64decode(image_data.img_body_base64)
        pil_image = Image.open(BytesIO(img_bytes))
        numpy_image = np.array(pil_image)
        
        # Teste verschiedene Vorverarbeitungen
        methods = [
            ('otsu', preprocess_image_v1(numpy_image)),
            ('adaptive', preprocess_image_v2(numpy_image))
        ]
        
        best_result = None
        best_score = -1
        all_results = []
        
        for method_name, preprocessed in methods:
            # Teste verschiedene PSM-Modi
            for psm in [6, 4, 3]:
                config = f'--oem 3 --psm {psm}'
                
                try:
                    ocr_text = pytesseract.image_to_string(
                        preprocessed,
                        lang='deu',
                        config=config
                    )
                    
                    result = extract_recipient_with_scoring(ocr_text)
                    result['method'] = f"{method_name}_psm{psm}"
                    result['ocr_text'] = ocr_text
                    
                    all_results.append(result)
                    
                    # Bester Score?
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_result = result
                except:
                    continue
        
        # Fallback
        if not best_result or best_score <= 0:
            best_result = all_results[0] if all_results else {
                "name": "Keinen Empfänger gefunden",
                "score": 0,
                "method": "none",
                "ocr_text": ""
            }
        
        return {
            "success": True,
            "message": "Bild erfolgreich verarbeitet",
            "image_size": len(img_bytes),
            "recipient": best_result['name'],
            "confidence_score": best_result['score'],
            "method_used": best_result.get('method', 'unknown'),
            "ocr_text": best_result.get('ocr_text', ''),
            "top_candidates": best_result.get('candidates', [])
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Fehler beim Verarbeiten des Bildes: {str(e)}"
        )