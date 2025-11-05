import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict


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