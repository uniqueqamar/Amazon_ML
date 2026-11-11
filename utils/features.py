"""
Pairwise similarity feature engineering for business entity matching.
Includes Levenshtein, Jaro-Winkler, token-based, numeric overlap, and country matching metrics.
"""
import math
import re
from typing import Dict, List, Set, Tuple
import numpy as np

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculates Levenshtein similarity ratio in [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(s1, s2) / 100.0

    len1, len2 = len(s1), len(s2)
    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i, c1 in enumerate(s1):
        curr[0] = i + 1
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
        prev, curr = curr, prev

    dist = prev[len2]
    return 1.0 - (dist / max(len1, len2))


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Calculates Jaro-Winkler similarity in [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return distance.JaroWinkler.similarity(s1, s2)

    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    m = matches
    jaro = (m / len1 + m / len2 + (m - transpositions / 2) / m) / 3.0

    prefix = 0
    for i in range(min(len1, len2, max_l)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * p * (1.0 - jaro)


def extract_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """Containment ratio: shared tokens divided by min token count."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    return intersection / min(len(set1), len(set2))


def token_sort_similarity(s1: str, s2: str) -> float:
    """Sorts tokens alphabetically and computes Levenshtein similarity."""
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return levenshtein_similarity(t1, t2)


def extract_numbers(text: str) -> Set[str]:
    """Extracts numeric strings (pincodes, street numbers, phone numbers)."""
    return set(re.findall(r'\b\d+\b', text or ""))


def compute_pair_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
    source_pair: str = "s1_s2",
    tfidf_score: float = 0.0
) -> Dict[str, float]:
    """
    Computes pairwise similarity features between a Source 1 entity and a Source 2/3 entity.
    """
    name1, name2 = name1 or "", name2 or ""
    addr1, addr2 = addr1 or "", addr2 or ""
    c1, c2 = (country1 or "").lower().strip(), (country2 or "").lower().strip()

    name_tokens1 = name1.split()
    name_tokens2 = name2.split()
    addr_tokens1 = addr1.split()
    addr_tokens2 = addr2.split()

    # Exact matches
    name_exact = 1.0 if name1 == name2 and name1 else 0.0
    addr_exact = 1.0 if addr1 == addr2 and addr1 else 0.0

    # Name similarity metrics
    name_lev = levenshtein_similarity(name1, name2)
    name_jw = jaro_winkler_similarity(name1, name2)
    name_token_sort = token_sort_similarity(name1, name2)
    name_token_jaccard = jaccard_similarity(set(name_tokens1), set(name_tokens2))
    name_token_overlap = token_overlap_ratio(name_tokens1, name_tokens2)
    name_ngram_jaccard = jaccard_similarity(extract_ngrams(name1, 3), extract_ngrams(name2, 3))

    # First char & first token matches
    first_char_match = 1.0 if (name1 and name2 and name1[0] == name2[0]) else 0.0
    first_token_match = 1.0 if (name_tokens1 and name_tokens2 and name_tokens1[0] == name_tokens2[0]) else 0.0

    # Length discrepancies
    name_len_diff = float(abs(len(name1) - len(name2)))
    name_len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)

    # Address similarity metrics
    addr_lev = levenshtein_similarity(addr1, addr2)
    addr_jw = jaro_winkler_similarity(addr1, addr2)
    addr_token_sort = token_sort_similarity(addr1, addr2)
    addr_token_jaccard = jaccard_similarity(set(addr_tokens1), set(addr_tokens2))
    addr_token_overlap = token_overlap_ratio(addr_tokens1, addr_tokens2)
    addr_ngram_jaccard = jaccard_similarity(extract_ngrams(addr1, 3), extract_ngrams(addr2, 3))

    addr_len_diff = float(abs(len(addr1) - len(addr2)))
    addr_len_ratio = min(len(addr1), len(addr2)) / max(len(addr1), len(addr2), 1)

    # Numeric match (crucial for street numbers, PIN codes)
    nums1 = extract_numbers(addr1) | extract_numbers(name1)
    nums2 = extract_numbers(addr2) | extract_numbers(name2)
    num_jaccard = jaccard_similarity(nums1, nums2)
    num_common = float(len(nums1.intersection(nums2)))

    # Country matching
    country_match = 1.0 if (c1 and c2 and c1 == c2) else (0.5 if not c1 or not c2 else 0.0)

    # Source pairing indicator
    is_s1_s3 = 1.0 if "s3" in source_pair.lower() else 0.0

    return {
        "name_exact": name_exact,
        "name_lev": name_lev,
        "name_jw": name_jw,
        "name_token_sort": name_token_sort,
        "name_token_jaccard": name_token_jaccard,
        "name_token_overlap": name_token_overlap,
        "name_ngram_jaccard": name_ngram_jaccard,
        "first_char_match": first_char_match,
        "first_token_match": first_token_match,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_exact": addr_exact,
        "addr_lev": addr_lev,
        "addr_jw": addr_jw,
        "addr_token_sort": addr_token_sort,
        "addr_token_jaccard": addr_token_jaccard,
        "addr_token_overlap": addr_token_overlap,
        "addr_ngram_jaccard": addr_ngram_jaccard,
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": addr_len_ratio,
        "num_jaccard": num_jaccard,
        "num_common": num_common,
        "country_match": country_match,
        "tfidf_score": float(tfidf_score),
        "is_s1_s3": is_s1_s3,
    }
