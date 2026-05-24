import os
import asyncio
import logging
import hashlib
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import edge_tts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model file paths relative to this file
_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(_dir, "voices-v1.0.bin")

# ---------------------------------------------------------------------------
# Curated Voice Catalog — 12 voices mapped to Kokoro embeddings
# ---------------------------------------------------------------------------
VOICE_CATALOG = {
    "aria": {"id": "af_bella", "name": "Aria", "gender": "Female", "accent": "American", "style": "Warm & Expressive"},
    "guy": {"id": "am_adam", "name": "Guy", "gender": "Male", "accent": "American", "style": "Confident & Clear"},
    "jenny": {"id": "af_sarah", "name": "Jenny", "gender": "Female", "accent": "American", "style": "Friendly & Casual"},
    "davis": {"id": "am_echo", "name": "Davis", "gender": "Male", "accent": "American", "style": "Deep & Authoritative"},
    "sonia": {"id": "bf_emma", "name": "Sonia", "gender": "Female", "accent": "British", "style": "Elegant & Refined"},
    "ryan": {"id": "bm_george", "name": "Ryan", "gender": "Male", "accent": "British", "style": "Warm & Natural"},
    "natasha": {"id": "bf_isabella", "name": "Natasha", "gender": "Female", "accent": "British", "style": "Bright & Energetic"},
    "william": {"id": "bm_lewis", "name": "William", "gender": "Male", "accent": "British", "style": "Calm & Steady"},
    "neerja": {"id": "af_sky", "name": "Neerja", "gender": "Female", "accent": "American", "style": "Clear & Melodic"},
    "prabhat": {"id": "am_eric", "name": "Prabhat", "gender": "Male", "accent": "American", "style": "Rich & Resonant"},
    "clara": {"id": "af_nicole", "name": "Clara", "gender": "Female", "accent": "American", "style": "Soft & Gentle"},
    "liam": {"id": "am_michael", "name": "Liam", "gender": "Male", "accent": "American", "style": "Friendly & Engaging"},
}

_FEMALE_KEYS = [k for k, v in VOICE_CATALOG.items() if v["gender"] == "Female"]
_MALE_KEYS = [k for k, v in VOICE_CATALOG.items() if v["gender"] == "Male"]

# Lazy-loaded Kokoro instance
_kokoro_instance = None

def get_kokoro():
    """Lazily load the Kokoro-ONNX InferenceSession, trying CUDA first."""
    global _kokoro_instance
    if _kokoro_instance is not None:
        return _kokoro_instance

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOICES_PATH):
        logger.error(f"Kokoro model files not found! Model: {MODEL_PATH}, Voices: {VOICES_PATH}")
        return None

    try:
        from kokoro_onnx import Kokoro
        from onnxruntime import InferenceSession
        
        providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        logger.info(f"Loading InferenceSession for Kokoro with providers: {providers}...")
        inf_sess = InferenceSession(MODEL_PATH, providers=providers)
        logger.info(f"Active ONNX Providers: {inf_sess.get_providers()}")
        _kokoro_instance = Kokoro.from_session(inf_sess, VOICES_PATH)
        logger.info("Kokoro-ONNX initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Kokoro: {e}")
        _kokoro_instance = None
    return _kokoro_instance


def _voice_key_for_speaker(speaker: str) -> str:
    """Fallback voice assignment using parser's gender check or simple hash parity."""
    try:
        from parser import detect_name_gender
        gender = detect_name_gender(speaker)
    except Exception:
        gender = "unknown"

    h = int(hashlib.md5(speaker.encode()).hexdigest(), 16)
    if gender == "female":
        pool = _FEMALE_KEYS
    elif gender == "male":
        pool = _MALE_KEYS
    else:
        pool = _FEMALE_KEYS if h % 2 == 0 else _MALE_KEYS
        
    return pool[h % len(pool)]


async def _generate_speech_edge_tts(text: str, output_path: str, voice_key: str, emotion: str):
    """
    Edge-TTS fallback voice generation.
    """
    # Map curated keys to Edge TTS voice IDs
    edge_voice_map = {
        "aria": "en-US-AriaNeural",
        "guy": "en-US-GuyNeural",
        "jenny": "en-US-JennyNeural",
        "davis": "en-US-GuyNeural",
        "sonia": "en-GB-SoniaNeural",
        "ryan": "en-GB-RyanNeural",
        "natasha": "en-AU-NatashaNeural",
        "william": "en-AU-WilliamMultilingualNeural",
        "neerja": "en-IN-NeerjaNeural",
        "prabhat": "en-IN-PrabhatNeural",
        "clara": "en-CA-ClaraNeural",
        "liam": "en-CA-LiamNeural",
    }
    
    # Try mapping Kokoro ids back to keys if key is a Kokoro ID
    reverse_catalog_map = {v["id"]: k for k, v in VOICE_CATALOG.items()}
    catalog_key = reverse_catalog_map.get(voice_key, voice_key)
    
    edge_voice = edge_voice_map.get(catalog_key, "en-US-GuyNeural")
    
    # Prosody speed / pitch shifts (subtle to avoid machinic sound)
    speed_params = {
        "neutral": "+0%", "joyful": "+8%", "sad": "-8%", "angry": "+5%",
        "terrified": "+12%", "whispering": "-12%", "shouting": "+10%",
        "monotone": "+0%", "trembling": "-5%", "authoritative": "+0%"
    }
    speed = speed_params.get(emotion.lower(), "+0%")
    
    communicate = edge_tts.Communicate(text=text, voice=edge_voice, rate=speed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    await communicate.save(output_path)


def generate_audio(
    text: str,
    speaker: str,
    emotion: str,
    output_path: str,
    voice_map: dict = None,
):
    """
    Main audio generation pipeline. Uses Kokoro-ONNX and falls back to Edge-TTS.
    """
    # Resolve voice key
    if voice_map and speaker in voice_map:
        voice_key = voice_map[speaker]
    elif speaker.lower() == "narrator":
        voice_key = "davis"
    else:
        voice_key = _voice_key_for_speaker(speaker)

    logger.info(
        f"Generating audio | Speaker: {speaker} | Resolved Voice: {voice_key} "
        f"| Emotion: {emotion} | Text: '{text[:40]}...'"
    )

    kokoro = get_kokoro()
    if kokoro:
        try:
            voice_info = VOICE_CATALOG.get(voice_key, VOICE_CATALOG["davis"])
            voice_id = voice_info["id"]

            # Emotion maps to speed factor in Kokoro
            speed_map = {
                "neutral": 1.0, "joyful": 1.05, "sad": 0.9, "angry": 1.05,
                "terrified": 1.1, "whispering": 0.85, "shouting": 1.05,
                "monotone": 1.0, "trembling": 0.9, "authoritative": 1.0
            }
            speed = speed_map.get(emotion.lower(), 1.0)
            
            # Generate local speech
            samples, sample_rate = kokoro.create(text, voice=voice_id, speed=speed, lang="en-us")
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # Save using soundfile to a temporary WAV, then convert using pydub
            temp_wav = output_path + ".temp.wav"
            sf.write(temp_wav, samples, sample_rate)
            
            audio = AudioSegment.from_wav(temp_wav)
            ext = os.path.splitext(output_path)[1].lower().lstrip('.')
            if not ext:
                ext = "mp3"
            
            audio.export(output_path, format=ext)
            
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as re:
                    logger.warning(f"Could not remove temp wav: {re}")
                    
            logger.info(f"Successfully generated Kokoro audio: {output_path}")
            return
        except Exception as ke:
            logger.error(f"Kokoro generation failed: {ke}. Falling back to Edge-TTS.")

    # Fallback to Edge-TTS
    try:
        asyncio.run(_generate_speech_edge_tts(text, output_path, voice_key, emotion))
        logger.info(f"Generated fallback Edge-TTS audio to {output_path}")
    except Exception as ee:
        logger.error(f"Fallback Edge-TTS failed: {ee}. Producing silent audio clip.")
        silent = AudioSegment.silent(duration=500)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fmt = os.path.splitext(output_path)[1].lower().lstrip('.') or "mp3"
        silent.export(output_path, format=fmt)


def generate_preview(voice_key: str, output_path: str):
    """
    Generate a short preview sample for a given voice key.
    """
    sample_text = (
        "Hello! This is a preview of my voice. "
        "I can narrate your stories with natural warmth and clarity."
    )
    logger.info(f"Generating preview for voice '{voice_key}' → {output_path}")
    generate_audio(sample_text, "Narrator", "neutral", output_path, voice_map={ "Narrator": voice_key })


def get_voice_catalog() -> dict:
    """Return the full voice catalog dictionary."""
    return VOICE_CATALOG


if __name__ == "__main__":
    test_out = "test_output.mp3"
    print("Running voice generator test...")
    try:
        generate_audio(
            text="This is a test of the upgraded high-fidelity narration engine.",
            speaker="Narrator",
            emotion="joyful",
            output_path=test_out,
        )
        if os.path.exists(test_out):
            print(f"Test Successful: {test_out} created.")
        else:
            print("Test Failed: Audio file not found.")
    except Exception as e:
        print(f"Test Failed with error: {e}")
