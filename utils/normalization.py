"""
Text and entity normalization utilities for Amazon ML Challenge entity resolution.
Handles business names, addresses, abbreviations, multilingual characters, and punctuation.
"""
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

import config

def remove_accents(text: str) -> str:
    """Normalize unicode characters and remove diacritics."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in text if not unicodedata.combining(c))

def normalize_text(text: Optional[str]) -> str:
    """General text normalization: lowercasing, whitespace trimming, and punctuation cleanup."""
    if text is None:
        return ""
    text = str(text)
    text = remove_accents(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def replace_abbreviations(text: str, mapping: Dict[str, str]) -> str:
    """Replace known abbreviations or synonyms with standardized forms using word boundaries."""
    if not text:
        return ""
    for abbr, full in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, full, text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_business_name(name: Optional[str]) -> str:
    """Normalize business entity names: standardizes legal suffixes and removes punctuation."""
    text = normalize_text(name)
    text = replace_abbreviations(text, config.BUSINESS_SYNONYMS)
    return text

def normalize_address(address: Optional[str]) -> str:
    """Normalize address fields: standardizes state codes and street markers."""
    text = normalize_text(address)
    text = replace_abbreviations(text, config.LOCATION_SYNONYMS)
    return text

def create_entity_representation(name: Optional[str], address: Optional[str], country: Optional[str] = "") -> str:
    """
    Combines normalized name, address, and country into a unified textual representation
    for TF-IDF indexing and vector search.
    """
    norm_name = normalize_business_name(name)
    norm_addr = normalize_address(address)
    norm_country = (country or "").lower().strip()

    parts = []
    if norm_name:
        parts.extend([norm_name, norm_name])  # Give name double weighting
    if norm_addr:
        parts.append(norm_addr)
    if norm_country:
        parts.append(norm_country)

    return " ".join(parts).strip()
