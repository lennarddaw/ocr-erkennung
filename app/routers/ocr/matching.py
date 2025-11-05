import re
import Levenshtein
from rapidfuzz import fuzz, process
from typing import List, Dict, Tuple, Optional, Set
from itertools import combinations, permutations


TITLE_PREFIXES = {'dr', 'prof', 'mr', 'mrs', 'ms', 'dr.', 'prof.'}


def normalize_word(word: str) -> Optional[str]:
    word_lower = word.lower()
    if word_lower in TITLE_PREFIXES:
        return None
    return word


def is_location_keyword(word: str, company_keywords: List[str], case_sensitive: bool, filter_enabled: bool) -> bool:
    if not filter_enabled:
        return False
    
    word_lower = word.lower()
    for keyword in company_keywords:
        if case_sensitive:
            if keyword in word:
                return True
        else:
            if keyword.lower() in word_lower:
                return True
    return False


def extract_capitalized_words(text: str, min_word_length: int, company_keywords: List[str], case_sensitive: bool, filter_enabled: bool) -> Set[str]:
    if not text or not text.strip():
        return set()
    
    word_pool = set()
    
    for line in text.split('\n'):
        clean = re.sub(r'[^a-zA-ZäöüßÄÖÜ\s]', ' ', line)
        words = [w.strip() for w in clean.split() if len(w.strip()) >= min_word_length]
        
        for word in words:
            if word and word[0].isupper():
                normalized = normalize_word(word)
                if normalized and not is_location_keyword(word, company_keywords, case_sensitive, filter_enabled):
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


def filter_by_locations(candidates: List[str], known_locations: List[str], location_threshold: int, filter_enabled: bool) -> List[str]:
    if not filter_enabled or not known_locations:
        return candidates
    
    filtered = []
    
    for candidate in candidates:
        match = process.extractOne(
            candidate,
            known_locations,
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] >= location_threshold:
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


def levenshtein_name_parts_match(word_pool: Set[str], recipients: List[str], name_parts_cache: Dict) -> Tuple[Optional[str], int, str]:
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
        if word_lower in name_parts_cache:
            possible_recipients = name_parts_cache[word_lower]
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


def match_word_pool_comprehensive(
    word_pool: Set[str],
    known_recipients: List[str],
    known_locations: List[str],
    company_keywords: List[str],
    location_threshold: int,
    filter_enabled: bool,
    fuzzy_threshold: int,
    enable_fallback: bool,
    name_parts_cache: Dict
) -> Dict:
    if not word_pool:
        return {
            "name": "No words found",
            "confidence": 0,
            "ocr_text": None,
            "method": "no_words",
            "matched_from_list": False
        }
    
    if not known_recipients:
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
    
    match, score, method = advanced_fuzzy_match(word_pool_string, known_recipients)
    if score > best_score:
        best_match = match
        best_score = score
        best_method = f"word_pool_{method}"
    
    combinations_list = create_smart_combinations(word_pool)
    filtered_combinations = filter_by_locations([c[0] for c in combinations_list], known_locations, location_threshold, filter_enabled)
    
    for candidate in filtered_combinations[:20]:
        match, score, method = advanced_fuzzy_match(candidate, known_recipients)
        if score > best_score:
            best_match = match
            best_score = score
            best_method = f"combination_{method}"
            print(f"Better: '{candidate}' -> '{best_match}' ({best_score}%, {best_method})")
    
    parts_match, parts_score, parts_method = levenshtein_name_parts_match(word_pool, known_recipients, name_parts_cache)
    if parts_score > best_score:
        best_match = parts_match
        best_score = parts_score
        best_method = parts_method
        print(f"Name parts: '{word_pool_string}' -> '{best_match}' ({best_score}%)")
    
    if best_match:
        print(f"Best: '{word_pool_string}' -> '{best_match}' ({best_score}%, {best_method})")
    
    adjusted_threshold = fuzzy_threshold - 5 if best_method and "levenshtein" in best_method else fuzzy_threshold
    
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
    elif enable_fallback and best_match and best_score >= 45:
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