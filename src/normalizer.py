"""
Text normalization for business names and addresses.

Design principles:
  - Keep BOTH original and normalized versions of every field.
  - Normalization is destructive on purpose — it makes different
    representations of the same entity look more alike.
  - Devanagari text is detected and flagged but NOT destroyed; the
    matching phase handles cross-script comparison separately.
  - French accented characters are NFKD-normalized but accents are
    preserved (they carry meaning in French).
"""

import re
import unicodedata
from typing import List, Optional, Set, Tuple

import pandas as pd

# ─── Compiled regexes (compiled once, reused millions of times) ──────────────

# Matches single-letter-dot patterns like A.B.C. or A. B. C.
_RE_INITIAL_DOTS = re.compile(r"\b([A-Za-z])\.(?=\s*[A-Za-z]\.|\s|$)")

# Matches common noise characters
_RE_NOISE = re.compile(r"[<>\-]{2,}|[/\\]{2,}|[~`!@#$%^*+={}|\[\]]")

# Matches multiple whitespace
_RE_MULTI_SPACE = re.compile(r"\s+")

# Matches domain extensions at end of name
_RE_DOMAIN = re.compile(
    r"\.(com|org|net|in|co\.in|co\.uk|io|biz|info|us|fr|de)$",
    re.IGNORECASE,
)

# Matches Devanagari script range
_RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Matches digits (for extracting numbers from addresses)
_RE_NUMBERS = re.compile(r"\b\d+\b")

# Matches PIN/ZIP codes: Indian 6-digit or US 5-digit (optionally +4)
_RE_PIN_ZIP = re.compile(r"\b(\d{6}|\d{5}(?:-\d{4})?)\b")

# ─── Legal suffix normalization ──────────────────────────────────────────────

# Order matters: longer patterns first to avoid partial matches.
# Each tuple is (compiled_regex, replacement_for_detection).
_LEGAL_SUFFIXES: List[Tuple[re.Pattern, str]] = []

_LEGAL_SUFFIX_PATTERNS = [
    # India
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bpvt\.?\s*ltd\.?\b", "pvt ltd"),
    (r"\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bllp\b", "llp"),
    (r"\blimited\b", "ltd"),
    (r"\bltd\.?\b", "ltd"),
    # US
    (r"\bincorporated\b", "inc"),
    (r"\binc\.?\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcorp\.?\b", "corp"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\bllc\b", "llc"),
    (r"\bco\.?\b", "co"),
    # France
    (r"\bsoci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b", "sarl"),
    (r"\bsarl\b", "sarl"),
    (r"\bsoci[eé]t[eé]\s+par\s+actions\s+simplifi[eé]e\b", "sas"),
    (r"\bsas\b", "sas"),
    (r"\bsoci[eé]t[eé]\s+anonyme\b", "sa"),
    (r"\bs\.?a\.?\b", "sa"),
    (r"\beurl\b", "eurl"),
    (r"\bsci\b", "sci"),
    # Germany (just in case)
    (r"\bgmbh\b", "gmbh"),
    (r"\bag\b", "ag"),
]

for pattern, label in _LEGAL_SUFFIX_PATTERNS:
    _LEGAL_SUFFIXES.append((re.compile(pattern, re.IGNORECASE), label))


# ─── Address abbreviation normalization ──────────────────────────────────────

_ADDR_ABBREVS = [
    (re.compile(r"\broad\b", re.IGNORECASE), "rd"),
    (re.compile(r"\bstreet\b", re.IGNORECASE), "st"),
    (re.compile(r"\bavenue\b", re.IGNORECASE), "ave"),
    (re.compile(r"\bboulevard\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\bdrive\b", re.IGNORECASE), "dr"),
    (re.compile(r"\bcourt\b", re.IGNORECASE), "ct"),
    (re.compile(r"\bplace\b", re.IGNORECASE), "pl"),
    (re.compile(r"\blane\b", re.IGNORECASE), "ln"),
    (re.compile(r"\bcircle\b", re.IGNORECASE), "cir"),
    (re.compile(r"\bhighway\b", re.IGNORECASE), "hwy"),
    (re.compile(r"\bparkway\b", re.IGNORECASE), "pkwy"),
    (re.compile(r"\bapartment\b", re.IGNORECASE), "apt"),
    (re.compile(r"\bsuite\b", re.IGNORECASE), "ste"),
    (re.compile(r"\bbuilding\b", re.IGNORECASE), "bldg"),
    (re.compile(r"\bfloor\b", re.IGNORECASE), "fl"),
    (re.compile(r"\bnumber\b", re.IGNORECASE), "no"),
    (re.compile(r"\bnagar\b", re.IGNORECASE), "ngr"),
    (re.compile(r"\bdistrict\b", re.IGNORECASE), "dist"),
    # French
    (re.compile(r"\brue\b", re.IGNORECASE), "rue"),
    (re.compile(r"\bav(?:enue)?\b", re.IGNORECASE), "ave"),
    (re.compile(r"\bbd\b", re.IGNORECASE), "blvd"),
]


# ═════════════════════════════════════════════════════════════════════════════
# Core normalization functions
# ═════════════════════════════════════════════════════════════════════════════

def is_devanagari(text: str) -> bool:
    """Return True if text contains Devanagari script characters."""
    return bool(_RE_DEVANAGARI.search(text))


def unicode_normalize(text: str) -> str:
    """Apply NFKC normalization — compatibility decomposition + canonical composition.

    NFKC (not NFKD) is used because NFKD decomposes accented characters
    into base + combining mark, and the later punctuation-strip regex
    would then destroy the combining marks (e.g. é → e + ́ → e).
    NFKC recomposes them so é stays as a single codepoint.
    """
    return unicodedata.normalize("NFKC", text)


def detect_legal_suffix(text: str) -> Optional[str]:
    """Detect the legal suffix type in a business name.

    Returns the normalized suffix label (e.g. "pvt ltd", "inc")
    or None if no legal suffix is found.
    """
    text_lower = text.lower()
    for regex, label in _LEGAL_SUFFIXES:
        if regex.search(text_lower):
            return label
    return None


def remove_legal_suffix(text: str) -> str:
    """Remove all legal suffix patterns from text."""
    for regex, _ in _LEGAL_SUFFIXES:
        text = regex.sub("", text)
    return text


def normalize_business_name(name: str) -> str:
    """Normalize a business name for comparison.

    Steps:
      1. Unicode NFKD normalization
      2. Lowercase
      3. Remove domain extensions (.com, .in, etc.)
      4. Remove dots between initials (A.B.C. → ABC)
      5. Remove legal suffixes
      6. Normalize "&" / "and"
      7. Strip noise characters (--, <<, >>, etc.)
      8. Collapse whitespace
      9. Strip
    """
    if not name:
        return ""

    # Skip heavy processing for Devanagari names — they'll be handled
    # separately via transliteration in the feature phase.
    if is_devanagari(name):
        # Still do basic cleanup
        name = _RE_MULTI_SPACE.sub(" ", name).strip()
        return name

    text = unicode_normalize(name)
    text = text.lower()

    # Remove domain extensions
    text = _RE_DOMAIN.sub("", text)

    # Remove dots between initials: A.B.C. → ABC
    text = _RE_INITIAL_DOTS.sub(r"\1", text)
    # Clean up remaining isolated dots (not decimal points)
    text = re.sub(r"(?<![0-9])\.(?![0-9])", " ", text)

    # Remove legal suffixes
    text = remove_legal_suffix(text)

    # Normalize "&" and "and"
    text = re.sub(r"\s*&\s*", " and ", text)

    # Strip noise characters
    text = _RE_NOISE.sub(" ", text)

    # Remove remaining punctuation except alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace and strip
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def normalize_address(address: str) -> str:
    """Normalize a business address for comparison.

    Steps:
      1. Unicode NFKD normalization
      2. Lowercase
      3. Normalize common abbreviations
      4. Remove punctuation (keep alphanumeric and spaces)
      5. Collapse whitespace
    """
    if not address:
        return ""

    text = unicode_normalize(address)
    text = text.lower()

    # Apply address abbreviation normalization
    for regex, replacement in _ADDR_ABBREVS:
        text = regex.sub(replacement, text)

    # Remove punctuation except alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse whitespace and strip
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def tokenize(text: str) -> Set[str]:
    """Split text into a set of non-empty lowercase tokens."""
    if not text:
        return set()
    return set(text.lower().split())


def extract_numbers(text: str) -> List[str]:
    """Extract all digit sequences from text."""
    if not text:
        return []
    return _RE_NUMBERS.findall(text)


def extract_pin_zip(text: str) -> Optional[str]:
    """Extract first PIN/ZIP code from text."""
    if not text:
        return None
    match = _RE_PIN_ZIP.search(text)
    return match.group(1) if match else None


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generate character n-grams from text."""
    if not text or len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


# ═════════════════════════════════════════════════════════════════════════════
# DataFrame-level normalization
# ═════════════════════════════════════════════════════════════════════════════

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns to a source DataFrame.

    Adds:
      - norm_name           : normalized business name
      - norm_name_tokens    : space-joined sorted tokens (for storage; convert to set at use)
      - has_legal_suffix    : 1 if a legal suffix was detected, else 0
      - legal_suffix_type   : the detected suffix type or ""
      - norm_address        : normalized address
      - norm_address_tokens : space-joined sorted address tokens
      - address_numbers     : comma-joined extracted numbers
      - pin_zip             : extracted PIN/ZIP or ""
      - is_devanagari       : 1 if name contains Devanagari script, else 0

    The original columns (business_name, business_address) are preserved.
    """
    df = df.copy()

    # ── Name normalization ────────────────────────────────────────────────
    df["is_devanagari"] = df["business_name"].apply(
        lambda x: 1 if is_devanagari(x) else 0
    ).astype("int8")

    df["legal_suffix_type"] = df["business_name"].apply(
        lambda x: detect_legal_suffix(x) or ""
    )
    df["has_legal_suffix"] = (df["legal_suffix_type"] != "").astype("int8")

    df["norm_name"] = df["business_name"].apply(normalize_business_name)

    df["norm_name_tokens"] = df["norm_name"].apply(
        lambda x: " ".join(sorted(tokenize(x)))
    )

    # ── Address normalization ─────────────────────────────────────────────
    df["norm_address"] = df["business_address"].apply(normalize_address)

    df["norm_address_tokens"] = df["norm_address"].apply(
        lambda x: " ".join(sorted(tokenize(x)))
    )

    df["address_numbers"] = df["business_address"].apply(
        lambda x: ",".join(extract_numbers(x))
    )

    df["pin_zip"] = df["business_address"].apply(
        lambda x: extract_pin_zip(x) or ""
    )

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Quick test
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test name normalization
    test_names = [
        "A.B.C. Restaurant Pvt. Ltd.",
        "ABC PRIVATE LIMITED",
        "abc rest.",
        "Smith & Sons Inc.",
        "wilfordhancock.com",
        "<< Team Ecole",
        "-- Holloway Peak Inc Seafood",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "Société Anonyme des Boulangeries",
        "LLC Moncada Léarning Center",
    ]

    print("═" * 70)
    print("  BUSINESS NAME NORMALIZATION TESTS")
    print("═" * 70)
    for name in test_names:
        norm = normalize_business_name(name)
        suffix = detect_legal_suffix(name)
        deva = is_devanagari(name)
        print(f"  Input : {name}")
        print(f"  Output: {norm}")
        print(f"  Suffix: {suffix}  Devanagari: {deva}")
        print(f"  Tokens: {sorted(tokenize(norm))}")
        print()

    # Test address normalization
    test_addrs = [
        "1795 Westchester Drive, High Point, NC",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux",
        "",
    ]

    print("═" * 70)
    print("  ADDRESS NORMALIZATION TESTS")
    print("═" * 70)
    for addr in test_addrs:
        norm = normalize_address(addr)
        nums = extract_numbers(addr)
        pin = extract_pin_zip(addr)
        print(f"  Input  : {addr}")
        print(f"  Output : {norm}")
        print(f"  Numbers: {nums}  PIN/ZIP: {pin}")
        print()
