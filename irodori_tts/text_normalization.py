from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Simple character-level replacements
# ---------------------------------------------------------------------------

SIMPLE_REPLACE_MAP: dict[str, str] = {
    # Whitespace / control
    "\t":     " ",
    "\r":     "",
    "\ufeff": "",       # BOM
    "\u200b": "",       # zero-width space
    "\u200c": "",       # zero-width non-joiner
    "\u200d": "",       # zero-width joiner
    "\u00ad": "",       # soft hyphen
    "\u00a0": " ",      # non-breaking space -> regular space

    # Full-width punctuation -> ASCII
    "？":     "?",
    "！":     "!",
    "，":     ",",
    "．":     ".",
    "：":     ":",
    "；":     ";",

    # Urdu-specific: keep native punctuation as-is
    # "،" (U+060C Arabic comma)    -> keep
    # "؟" (U+061F Arabic question) -> keep
    # "۔" (U+06D4 Urdu full stop)  -> keep

    # Common ASCII quote normalisation
    "\u2018": "'",      # '  left single quotation
    "\u2019": "'",      # '  right single quotation
    "\u201a": "'",      # ‚  single low quotation
    "\u201b": "'",      # ‛  single high-reversed quotation
    "\u201c": '"',      # "  left double quotation
    "\u201d": '"',      # "  right double quotation
    "\u201e": '"',      # „  double low quotation
    "\u201f": '"',      # ‟  double high-reversed quotation
    "\u00ab": '"',      # «  left-pointing double angle
    "\u00bb": '"',      # »  right-pointing double angle
    "\u2039": "'",      # ‹  left single angle
    "\u203a": "'",      # ›  right single angle

    # Dashes -> plain hyphen-minus
    "\u2012": "-",      # ‒  figure dash
    "\u2013": "-",      # –  en dash
    "\u2014": "-",      # —  em dash
    "\u2015": "-",      # ―  horizontal bar
    "\u2212": "-",      # −  minus sign
}

# ---------------------------------------------------------------------------
# Regex-level replacements
# ---------------------------------------------------------------------------

REGEX_REPLACE_MAP: dict[re.Pattern, str] = {
    # Strip decorative / technical symbols that carry no linguistic content
    re.compile(
        r"[▼▽△▲►◄♀♂《》≪≫"
        r"①②③④⑤⑥⑦⑧⑨⑩"
        r"©®™°•·]"
    ): "",

    # Strip Arabic ornamental / Quranic marks not used in Urdu prose
    re.compile(
        r"[\u0600-\u0605"       # Arabic number signs
        r"\u0609\u060A"         # per-mille / per-ten-thousand
        r"\u061B"               # Arabic semicolon
        r"\u061E"               # Arabic triple-dot punctuation mark
        r"\u06DD"               # Arabic end of Ayah
        r"\u06DE"               # Arabic start of Rub el Hizb
        r"\uFD3E\uFD3F]"        # ornate left/right parenthesis
    ): "",

    # Strip Quranic-only / ornamental Arabic marks NOT used in standard Urdu.
    #
    # KEPT (standard Urdu tashkeel — zabr/zer/pesh/shadda/sukun/tanwin):
    #   U+064B  ً  fathatan   (tanwin fath / zabr double)
    #   U+064C  ٌ  dammatan   (tanwin damm / pesh double)
    #   U+064D  ٍ  kasratan   (tanwin kasr / zer double)
    #   U+064E  َ  fathah     (zabr)
    #   U+064F  ُ  dammah     (pesh)
    #   U+0650  ِ  kasrah     (zer)
    #   U+0651  ّ  shadda     (tashdid — gemination)
    #   U+0652  ْ  sukun      (jazm — no vowel)
    #   U+0670  ٰ  superscript alef  (khari zabar)
    #
    # STRIPPED (Quranic recitation marks / ornamental signs only):
    #   U+0610–U+061A  Arabic religious/honorific signs
    #   U+0653–U+065F  Extended combining marks (maddah, hamza variants, etc.)
    #   U+06D6–U+06DC  Quranic small high ligatures
    #   U+06DF–U+06E4  Quranic pause marks
    #   U+06E7–U+06E8  Quranic small high yeh/noon
    #   U+06EA–U+06ED  Quranic stop marks
    re.compile(
        r"[\u0610-\u061A"       # Arabic honorific/religious signs (Sallallahou etc.)
        r"\u0653-\u065F"        # extended combining: maddah, hamza above/below, etc.
        r"\u06D6-\u06DC"        # Quranic small high ligatures
        r"\u06DF-\u06E4"        # Quranic pause / sajda marks
        r"\u06E7\u06E8"         # Quranic small high yeh / noon
        r"\u06EA-\u06ED]"       # Quranic stop marks
    ): "",

    # Strip box-drawing, arrows, and misc technical symbols
    re.compile(
        r"[\u02d7"              # modifier letter minus sign
        r"\u2010-\u2011"        # hyphen, non-breaking hyphen (already handled above but safe)
        r"\u2043"               # hyphen bullet
        r"\u23af\u23e4"         # horizontal line extension
        r"\u2500\u2501"         # box drawings
        r"\u2e3a\u2e3b]"        # two/three-em dash
    ): "",

    # Arabic-Indic digits -> Western Arabic digits
    re.compile(r"[٠١٢٣٤٥٦٧٨٩]"): lambda m: str(ord(m.group()) - 0x0660),

    # Extended Arabic-Indic digits (Perso-Arabic) -> Western
    re.compile(r"[۰۱۲۳۴۵۶۷۸۹]"): lambda m: str(ord(m.group()) - 0x06F0),

    # Collapse 3+ dots or Urdu stops into canonical ellipsis …
    re.compile(r"\.{2,}"): "…",
    re.compile(r"۔{2,}"): "…",

    # Collapse multiple exclamation/question marks
    re.compile(r"!{2,}"): "!",
    re.compile(r"\?{2,}"): "?",

    # Collapse runs of dashes longer than 2 into a single em-dash surrogate
    re.compile(r"-{3,}"): "--",
}

# ---------------------------------------------------------------------------
# Outer bracket stripping
# Mirrors the original strip_outer_brackets but adds Urdu/Arabic pairs
# ---------------------------------------------------------------------------

_BRACKET_PAIRS: dict[str, str] = {
    # ASCII / Latin
    "(":  ")",
    "[":  "]",
    "{":  "}",
    # Urdu / Arabic
    "«":  "»",
    # Fullwidth
    "（": "）",
    "【": "】",
    "「": "」",
    "『": "』",
    "《": "》",
    "〈": "〉",
}


def strip_outer_brackets(text: str) -> str:
    """
    Repeatedly strip a matching open/close bracket pair that wraps the
    entire string.  E.g. '(hello world)' -> 'hello world'.
    Handles nested brackets correctly — only strips when one pair wraps all.
    """
    while True:
        if len(text) < 2:
            break

        start_char = text[0]
        end_char   = text[-1]

        if start_char not in _BRACKET_PAIRS or _BRACKET_PAIRS[start_char] != end_char:
            break

        depth = 0
        is_enclosing_all = True

        for i, char in enumerate(text):
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
            if depth == 0 and i < len(text) - 1:
                is_enclosing_all = False
                break

        if is_enclosing_all and depth == 0:
            text = text[1:-1]
            continue

        break

    return text


# ---------------------------------------------------------------------------
# Whitespace normalisation
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE    = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE  = re.compile(r"\n{3,}")


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalise an Urdu or English transcript string.

    Steps (mirrors the original Japanese normalize_text pipeline):
      1. NFKC Unicode normalisation  (must come first so composed forms are
         split before substitutions run — avoids NFKC decomposing … back to ...)
      2. Simple character replacements (SIMPLE_REPLACE_MAP)
      3. Regex replacements (REGEX_REPLACE_MAP)
         - strip decorative symbols
         - strip Arabic ornamental / Quranic marks
         - strip tashkeel diacritics
         - convert Arabic-Indic / Perso-Arabic digits to Western
         - collapse repeated punctuation
         - collapse 2+ dots or Urdu full-stops -> …
      4. Strip enclosing bracket pair if it wraps the whole string
      5. Whitespace collapse + strip
    """
    if not text:
        return ""

    # 1. NFKC first — normalise composed Unicode forms
    text = unicodedata.normalize("NFKC", text)

    # 2. Simple replacements
    for old, new in SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)

    # 3. Regex replacements
    for pattern, replacement in REGEX_REPLACE_MAP.items():
        text = pattern.sub(replacement, text)

    # 4. Strip outer brackets
    text = strip_outer_brackets(text)

    # 5. Whitespace
    text = _normalize_whitespace(text)

    return text