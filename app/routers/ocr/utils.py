from PIL import Image
from pathlib import Path
import json
from typing import Tuple, List, Dict


def convert_heic_to_rgb(pil_image: Image.Image) -> Image.Image:
    try:
        if pil_image.format in ['HEIF', 'HEIC']:
            pil_image = pil_image.convert('RGB')
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        return pil_image.convert('RGB')


def load_recipients(recipients_file: Path) -> Tuple[List[str], Dict]:
    try:
        with open(recipients_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            recipients = data.get("recipients", [])
            settings = data.get("settings", {})
            print(f"Loaded {len(recipients)} recipients")
            return recipients, settings
    except Exception as e:
        print(f"Error loading recipients: {e}")
        return [], {}


def load_locations(locations_file: Path) -> Tuple[List[str], List[str], Dict]:
    try:
        with open(locations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            locations = data.get("locations", [])
            keywords = data.get("company_keywords", [])
            settings = data.get("settings", {})
            print(f"Loaded {len(locations)} locations, {len(keywords)} keywords")
            return locations, keywords, settings
    except Exception as e:
        print(f"Error loading locations: {e}")
        return [], [], {"filter_enabled": False}


def build_recipient_caches(recipients: List[str]) -> Tuple[Dict, Dict]:
    recipient_cache = {r.lower(): r for r in recipients}
    name_parts_cache = {}
    
    for recipient in recipients:
        parts = recipient.split()
        for part in parts:
            if len(part) >= 2:
                part_lower = part.lower()
                if part_lower not in name_parts_cache:
                    name_parts_cache[part_lower] = []
                name_parts_cache[part_lower].append(recipient)
    
    return recipient_cache, name_parts_cache