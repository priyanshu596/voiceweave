import re
import json
import time
import logging
import hashlib
from typing import List, Dict

# Lists of common names for classification fallback
FEMALE_NAMES = {
    "Aria", "Bella", "Sarah", "Sky", "Nicole", "Emma", "Isabella", "Clara", "Elena",
    "Natasha", "Sonia", "Jenny", "Neerja", "Jane", "Mary", "Elizabeth", "Alice",
    "Lucy", "Emily", "Charlotte", "Sophia", "Olivia", "Ava", "Mia", "Evelyn",
    "Harper", "Camila", "Gianna", "Abigail", "Luna", "Ella", "Avery", "Scarlett"
}

MALE_NAMES = {
    "Adam", "Michael", "Echo", "Eric", "George", "Lewis", "Guy", "Davis", "Ryan",
    "William", "Prabhat", "Liam", "John", "David", "Robert", "James", "Charles",
    "Thomas", "Arthur", "Julian", "Noah", "Oliver", "Elijah", "Benjamin", "Lucas",
    "Henry", "Alexander", "Mason", "Ethan", "Daniel", "Jacob", "Logan", "Jackson",
    "Levi", "Sebastian", "Jack"
}

def detect_name_gender(name: str) -> str:
    """Return 'female' or 'male' based on known lists of names, or 'unknown'."""
    name_clean = name.strip().lower()
    
    # Check known female names
    for female_name in FEMALE_NAMES:
        if female_name.lower() in name_clean:
            return "female"
            
    # Check known male names
    for male_name in MALE_NAMES:
        if male_name.lower() in name_clean:
            return "male"
            
    return "unknown"

def detect_gender_from_context(speaker: str, before: str, after: str) -> str:
    """Detect speaker gender from name lists and surrounding context pronouns."""
    if speaker.lower() == "narrator":
        return "male"
        
    gender = detect_name_gender(speaker)
    if gender != "unknown":
        return gender
        
    # Cut context at sentence boundaries/newlines just like in speaker extraction
    after_segment = ""
    if after:
        after_first_line = after.split('\n')[0]
        sentence_end = re.search(r"[.!?]", after_first_line)
        if sentence_end:
            after_segment = after_first_line[:sentence_end.start() + 1]
        else:
            after_segment = after_first_line
            
    before_segment = ""
    if before:
        before_last_line = before.split('\n')[-1]
        sentences = re.split(r"[.!?]", before_last_line)
        sentences = [s.strip() for s in sentences if s.strip()]
        before_segment = sentences[-1] if sentences else before_last_line

    context = (before_segment + " " + after_segment).lower()
    has_female = bool(re.search(r"\b(she|her|herself)\b", context))
    has_male = bool(re.search(r"\b(he|him|his|himself)\b", context))
    
    if has_female and not has_male:
        return "female"
    elif has_male and not has_female:
        return "male"
        
    # Hash fallback to keep it deterministic
    h = int(hashlib.md5(speaker.encode()).hexdigest(), 16)
    return "female" if h % 2 == 0 else "male"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RATE_LIMIT_DELAY = 0  # No API rate-limiting needed for rule-based parsing
MAX_CHUNKS = 8

# ---------------------------------------------------------------------------
# Emotion keyword → (tone, emotion) mapping
# ---------------------------------------------------------------------------
EMOTION_KEYWORDS = {
    # Original keywords
    "whispered": ("whispering", "whispering"),
    "whisper": ("whispering", "whispering"),
    "shouted": ("shouting", "angry"),
    "shout": ("shouting", "angry"),
    "yelled": ("shouting", "angry"),
    "yell": ("shouting", "angry"),
    "screamed": ("shouting", "terrified"),
    "scream": ("shouting", "terrified"),
    "cried": ("trembling", "sad"),
    "cry": ("trembling", "sad"),
    "sobbed": ("trembling", "sad"),
    "sob": ("trembling", "sad"),
    "murmured": ("whispering", "whispering"),
    "murmur": ("whispering", "whispering"),
    "muttered": ("whispering", "whispering"),
    "mutter": ("whispering", "whispering"),
    "laughed": ("monotone", "joyful"),
    "laugh": ("monotone", "joyful"),
    "giggled": ("monotone", "joyful"),
    "giggle": ("monotone", "joyful"),
    "sighed": ("monotone", "sad"),
    "sigh": ("monotone", "sad"),
    "growled": ("authoritative", "angry"),
    "growl": ("authoritative", "angry"),
    "snapped": ("authoritative", "angry"),
    "snap": ("authoritative", "angry"),
    "pleaded": ("trembling", "trembling"),
    "plead": ("trembling", "trembling"),
    "begged": ("trembling", "trembling"),
    "beg": ("trembling", "trembling"),
    "exclaimed": ("authoritative", "joyful"),
    "exclaim": ("authoritative", "joyful"),
    # Extended keywords
    "stammered": ("trembling", "trembling"),
    "stammer": ("trembling", "trembling"),
    "roared": ("shouting", "angry"),
    "roar": ("shouting", "angry"),
    "hissed": ("whispering", "angry"),
    "hiss": ("whispering", "angry"),
    "demanded": ("authoritative", "angry"),
    "demand": ("authoritative", "angry"),
    "insisted": ("authoritative", "authoritative"),
    "insist": ("authoritative", "authoritative"),
    "warned": ("authoritative", "terrified"),
    "warn": ("authoritative", "terrified"),
    "called": ("shouting", "neutral"),
    "call": ("shouting", "neutral"),
    "declared": ("authoritative", "authoritative"),
    "declare": ("authoritative", "authoritative"),
    "remarked": ("monotone", "neutral"),
    "remark": ("monotone", "neutral"),
    "interrupted": ("authoritative", "angry"),
    "interrupt": ("authoritative", "angry"),
    "wondered": ("whispering", "neutral"),
    "wonder": ("whispering", "neutral"),
    "chuckled": ("monotone", "joyful"),
    "chuckle": ("monotone", "joyful"),
    "wept": ("trembling", "sad"),
    "weep": ("trembling", "sad"),
    "groaned": ("monotone", "sad"),
    "groan": ("monotone", "sad"),
    "gasped": ("trembling", "terrified"),
    "gasp": ("trembling", "terrified"),
}

# Compile a regex of speech-verb keywords for attribution detection
_SPEECH_VERBS = sorted(EMOTION_KEYWORDS.keys(), key=len, reverse=True)
_SPEECH_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _SPEECH_VERBS) + r"|said|asked|replied|answered|told|spoke)\b",
    re.IGNORECASE,
)

# Pattern to extract a capitalised speaker name near a speech verb
_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]{1,15}(?:\s[A-Z][a-z]{1,15})?)\b")


def _normalize_quotes(text: str) -> str:
    """Replace smart / curly quotes with straight ASCII equivalents."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    """Split text into chunks that respect paragraph boundaries."""
    paragraphs = text.split('\n')
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + '\n'
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?]) +', para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) < max_chars:
                        sub += s + " "
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = s + " "
                current = sub
            else:
                current = para + '\n'
    if current:
        chunks.append(current.strip())
    return chunks


def _detect_emotion_from_narration(text: str) -> tuple:
    """Scan narration text for emotion keywords and return (emotion, tone)."""
    lower = text.lower()
    for keyword, (tone, emotion) in EMOTION_KEYWORDS.items():
        if keyword in lower:
            return emotion, tone
    return "neutral", "calm"


def _extract_speaker_from_context(before: str, after: str) -> str:
    """
    Try to find the speaker name from the narration surrounding a dialogue
    segment. Handles patterns like:
      - 'John said, "…"'        (speaker before)
      - '"…" said John'         (speaker after)
      - '"…" she whispered'     (pronoun — return generic)
    """
    # Check narration AFTER the quote: '"text" said John' / '"text" John said'
    if after:
        # Cut at first newline or sentence boundary to avoid leaking to next line/dialogue
        after_first_line = after.split('\n')[0]
        sentence_end = re.search(r"[.!?]", after_first_line)
        if sentence_end:
            after_trimmed = after_first_line[:sentence_end.start() + 1].lstrip(" ,.")
        else:
            after_trimmed = after_first_line.lstrip(" ,.")
            
        match = _SPEECH_VERB_PATTERN.search(after_trimmed[:80])
        if match:
            # Look for a capitalised name near the verb
            name_search = _NAME_PATTERN.search(after_trimmed[:80])
            if name_search:
                candidate = name_search.group(1)
                # Filter out common non-name words
                if candidate.lower() not in {
                    "the", "but", "and", "she", "her", "his",
                    "he", "they", "them", "it", "its", "this",
                    "that", "with", "from", "then", "there",
                    "here", "what", "when", "where", "who",
                    "how", "not", "all", "just", "only",
                }:
                    return candidate

    # Check narration BEFORE the quote: 'John said, "text"'
    if before:
        # Cut to get only the last sentence/clause before the quote on the last line
        before_last_line = before.split('\n')[-1]
        sentences = re.split(r"[.!?]", before_last_line)
        # Filter empty segments
        sentences = [s.strip() for s in sentences if s.strip()]
        before_trimmed = sentences[-1].rstrip(" ,.") if sentences else before_last_line.rstrip(" ,.")
        
        match = _SPEECH_VERB_PATTERN.search(before_trimmed[-80:])
        if match:
            name_search = _NAME_PATTERN.search(before_trimmed[-80:])
            if name_search:
                candidate = name_search.group(1)
                if candidate.lower() not in {
                    "the", "but", "and", "she", "her", "his",
                    "he", "they", "them", "it", "its", "this",
                    "that", "with", "from", "then", "there",
                    "here", "what", "when", "where", "who",
                    "how", "not", "all", "just", "only",
                }:
                    return candidate

    return "Character"


def rule_based_parse(text: str) -> List[Dict]:
    """
    Primary rule-based story parser.
    Splits text on quoted dialogue, detects speakers from surrounding
    narration, and infers emotions from speech-verb keywords.
    """
    logger.info("Parsing text with rule-based parser.")
    text = _normalize_quotes(text)

    blocks: List[Dict] = []

    # Split on quoted strings, keeping the quotes as capture groups
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)

    for idx, part in enumerate(parts):
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if part_stripped.startswith('"') and part_stripped.endswith('"'):
            # Dialogue block
            dialogue_text = part_stripped[1:-1]

            # Gather surrounding narration for speaker detection
            before_text = parts[idx - 1] if idx > 0 else ""
            after_text = parts[idx + 1] if idx < len(parts) - 1 else ""

            speaker = _extract_speaker_from_context(before_text, after_text)
            gender = detect_gender_from_context(speaker, before_text, after_text)

            # Detect emotion from the surrounding narration
            context = (before_text + " " + after_text).strip()
            emotion, tone = _detect_emotion_from_narration(context)

            blocks.append({
                "speaker": speaker,
                "dialogue": dialogue_text,
                "gender": gender,
                "emotion": emotion,
                "tone": tone,
            })
        else:
            # Narration block
            emotion, tone = _detect_emotion_from_narration(part_stripped)
            blocks.append({
                "speaker": "Narrator",
                "dialogue": part_stripped,
                "gender": "male",
                "emotion": emotion,
                "tone": tone,
            })

    return blocks


def get_detected_speakers(blocks: List[Dict]) -> List[str]:
    """Return a sorted list of unique speaker names from parsed blocks."""
    speakers = set()
    for block in blocks:
        speakers.add(block.get("speaker", "Narrator"))
    return sorted(speakers)


def get_detected_speakers_with_genders(blocks: List[Dict]) -> Dict[str, str]:
    """Return a mapping of speaker name to their detected gender."""
    mapping = {}
    for block in blocks:
        sp = block.get("speaker", "Narrator")
        gender = block.get("gender", "male")
        if sp not in mapping:
            mapping[sp] = gender
    return mapping



def parse_story(text: str) -> List[Dict]:
    """
    Parse a story / text passage into structured narration blocks.
    Uses the rule-based parser directly (no external API).
    """
    if not text or not text.strip():
        return []
    return rule_based_parse(text)


def parse_novel(text: str) -> List[Dict]:
    """Parse a longer text by chunking first, then parsing each chunk."""
    chunks = chunk_text(text)
    if len(chunks) > MAX_CHUNKS:
        logger.warning(f"Text produces {len(chunks)} chunks, truncating to {MAX_CHUNKS}")
        chunks = chunks[:MAX_CHUNKS]
    full: List[Dict] = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Parsing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        data = parse_story(chunk)
        full.extend(data)
    return full


if __name__ == "__main__":
    sample = '"No... don\'t leave me," she whispered.\n"Enough!" he shouted.\nJohn said, "We need to go now."'
    print("Parsing sample text...")
    result = parse_story(sample)
    print(json.dumps(result, indent=2))
    print("Detected speakers:", get_detected_speakers(result))
