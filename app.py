import os
import sys

# Dynamic CUDA/cuDNN relaunch helper for Linux GPU acceleration
if sys.platform.startswith("linux"):
    venv_site = os.path.join(os.getcwd(), "venv/lib/python3.10/site-packages")
    nvidia_paths = [
        os.path.join(venv_site, "nvidia/cuda_nvrtc/lib"),
        os.path.join(venv_site, "nvidia/cublas/lib"),
        os.path.join(venv_site, "nvidia/cudnn/lib"),
        os.path.join(venv_site, "nvidia/cuda_runtime/lib"),
    ]
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    missing = [p for p in nvidia_paths if p not in current_ld and os.path.exists(p)]
    
    if missing:
        new_ld = ":".join(nvidia_paths) + (":" + current_ld if current_ld else "")
        os.environ["LD_LIBRARY_PATH"] = new_ld
        try:
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
        except Exception as e:
            print(f"Warning: Failed to relaunch with GPU paths: {e}", file=sys.stderr)

import uuid
import json
import time
import logging
import threading

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

import PyPDF2
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from parser import parse_story, parse_novel, chunk_text, get_detected_speakers
from tts_engine import generate_audio, generate_preview, get_voice_catalog
from audio_mixer import merge_audio_clips

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_TEXT_LENGTH = 30000
RATE_LIMIT_DELAY = 0  # No API rate-limiting needed

jobs = {}


# ---------------------------------------------------------------------------
# File text extraction
# ---------------------------------------------------------------------------

def extract_text(file_path, max_length=MAX_TEXT_LENGTH):
    """Extract plain text from .txt, .pdf, or .epub files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    if len(text) >= max_length:
                        break
    elif ext == '.epub':
        book = epub.read_epub(file_path)
        text_parts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_parts.append(soup.get_text())
                combined = "\n".join(text_parts)
                if len(combined) >= max_length:
                    text_parts = [combined[:max_length]]
                    break
        text = "\n".join(text_parts)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if len(text) > max_length:
        text = text[:max_length]
        logger.info(f"Text truncated to {max_length} characters")
    return text


# ---------------------------------------------------------------------------
# Background novel processing
# ---------------------------------------------------------------------------

def process_novel(job_id, file_path, voice_map=None):
    """Process an uploaded novel file in a background thread."""
    try:
        jobs[job_id]['status'] = 'extracting'
        logger.info(f"Job {job_id}: Extracting text from {file_path}")
        text = extract_text(file_path)

        if not text.strip():
            raise ValueError("No text could be extracted from the file.")

        jobs[job_id]['status'] = 'chunking'
        chunks = chunk_text(text)
        total_chunks = len(chunks)
        jobs[job_id]['total_chunks'] = total_chunks
        jobs[job_id]['text_length'] = len(text)

        all_temp_clips = []
        all_speakers = []
        aborted_early = False
        abort_reason = None

        try:
            for i, chunk in enumerate(chunks):
                if aborted_early:
                    break
                jobs[job_id]['status'] = f'processing chunk {i+1}/{total_chunks}'
                jobs[job_id]['current_chunk'] = i + 1
                logger.info(f"Job {job_id}: Processing chunk {i+1}/{total_chunks} ({len(chunk)} chars)")

                parsed_blocks = parse_story(chunk)
                jobs[job_id]['chunk_total_clips'] = len(parsed_blocks)

                for j, block in enumerate(parsed_blocks):
                    jobs[job_id]['chunk_current_clip'] = j + 1
                    clip_filename = f"{job_id}_chunk{i}_clip{j}.mp3"
                    clip_path = os.path.join(OUTPUT_DIR, clip_filename)

                    try:
                        generate_audio(
                            text=block['dialogue'],
                            speaker=block['speaker'],
                            emotion=block['emotion'],
                            output_path=clip_path,
                            voice_map=voice_map,
                        )
                        all_temp_clips.append(clip_path)
                        all_speakers.append(block['speaker'])
                    except Exception as ge:
                        logger.warning(f"Job {job_id}: Generation failed/timed out on chunk {i} block {j}: {ge}. Aborting generation loop and saving completed parts.")
                        aborted_early = True
                        abort_reason = str(ge)
                        break
        except Exception as loop_err:
            logger.warning(f"Job {job_id}: Loop error encountered: {loop_err}. Proceeding with merged partial output if available.")
            aborted_early = True
            abort_reason = str(loop_err)

        if not all_temp_clips:
            raise ValueError(f"Failed to generate any audio. Reason: {abort_reason or 'No dialogue blocks found.'}")

        jobs[job_id]['status'] = 'merging'
        logger.info(f"Job {job_id}: Merging {len(all_temp_clips)} clips")
        final_filename = f"{job_id}_final.mp3"
        final_path = os.path.join(OUTPUT_DIR, final_filename)

        merge_audio_clips(all_temp_clips, final_path, speakers=all_speakers)

        for clip in all_temp_clips:
            try:
                os.remove(clip)
            except Exception as e:
                logger.warning(f"Failed to remove temporary clip {clip}: {e}")

        if aborted_early:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['audio_url'] = f"/outputs/{final_filename}"
            jobs[job_id]['warning'] = f"Completed partially. Aborted early due to: {abort_reason}"
            logger.info(f"Job {job_id}: Completed partially due to abort")
        else:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['audio_url'] = f"/outputs/{final_filename}"
            logger.info(f"Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


# ---------------------------------------------------------------------------
# Routes — Static / Audio
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/outputs/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ---------------------------------------------------------------------------
# Routes — Voice Catalog
# ---------------------------------------------------------------------------

@app.route('/api/voices', methods=['GET'])
def voices():
    """Return the full voice catalog."""
    catalog = get_voice_catalog()
    return jsonify(catalog)


@app.route('/api/voices/<voice_key>/preview', methods=['GET'])
def voice_preview(voice_key):
    """Generate and serve a short preview clip for a voice."""
    catalog = get_voice_catalog()
    if voice_key not in catalog:
        return jsonify({"error": f"Unknown voice key: {voice_key}"}), 404

    preview_filename = f"preview_{voice_key}.mp3"
    preview_path = os.path.join(OUTPUT_DIR, preview_filename)

    try:
        generate_preview(voice_key, preview_path)
        return send_from_directory(OUTPUT_DIR, preview_filename, mimetype="audio/mpeg")
    except Exception as e:
        logger.error(f"Preview generation failed for '{voice_key}': {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — Analyze File (Extract characters from uploaded document)
# ---------------------------------------------------------------------------

@app.route('/api/analyze-file', methods=['POST'])
def analyze_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.txt', '.pdf', '.epub']:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400
        
    temp_id = str(uuid.uuid4())
    filename = f"temp_analyze_{temp_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)
    
    try:
        text = extract_text(file_path)
        if not text.strip():
            return jsonify({"error": "Empty text extracted"}), 400
            
        # Parse a snippet of the text to detect characters (up to 12000 chars is plenty)
        sample_text = text[:12000]
        parsed_blocks = parse_story(sample_text)
        speakers = get_detected_speakers(parsed_blocks)
        from parser import get_detected_speakers_with_genders
        speaker_genders = get_detected_speakers_with_genders(parsed_blocks)
        
        return jsonify({
            "status": "success",
            "speakers": speakers,
            "speaker_genders": speaker_genders,
            "text_preview": text[:500] + "..."
        })
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as re:
                logger.warning(f"Failed to remove temp file {file_path}: {re}")


# ---------------------------------------------------------------------------
# Routes — Parse (Analyze text without generating audio)
# ---------------------------------------------------------------------------

@app.route('/api/parse', methods=['POST'])
def parse_only():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']
    if not text.strip():
        return jsonify({"error": "Empty text provided"}), 400
    try:
        parsed_blocks = parse_story(text)
        speakers = get_detected_speakers(parsed_blocks)
        from parser import get_detected_speakers_with_genders
        speaker_genders = get_detected_speakers_with_genders(parsed_blocks)
        return jsonify({
            "status": "success",
            "blocks": parsed_blocks,
            "speakers": speakers,
            "speaker_genders": speaker_genders
        })
    except Exception as e:
        logger.error(f"Error parsing story: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — Narrate (short text)
# ---------------------------------------------------------------------------

@app.route('/api/narrate', methods=['POST'])
def narrate():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    if not text.strip():
        return jsonify({"error": "Empty text provided"}), 400

    if len(text) > MAX_TEXT_LENGTH:
        logger.info(f"Input text truncated from {len(text)} to {MAX_TEXT_LENGTH} chars")
        text = text[:MAX_TEXT_LENGTH]

    voice_map = data.get('voice_map', None)

    try:
        logger.info("Parsing story...")
        parsed_blocks = parse_story(text)
        if not parsed_blocks:
            return jsonify({"error": "Failed to parse story"}), 500

        speakers = get_detected_speakers(parsed_blocks)

        request_id = str(uuid.uuid4())
        temp_clips = []
        clip_speakers = []

        logger.info(f"Generating audio for {len(parsed_blocks)} blocks (Request ID: {request_id})...")

        for i, block in enumerate(parsed_blocks):
            clip_filename = f"{request_id}_clip_{i}.mp3"
            clip_path = os.path.join(OUTPUT_DIR, clip_filename)

            generate_audio(
                text=block['dialogue'],
                speaker=block['speaker'],
                emotion=block['emotion'],
                output_path=clip_path,
                voice_map=voice_map,
            )
            temp_clips.append(clip_path)
            clip_speakers.append(block['speaker'])

        final_filename = f"{request_id}_final.mp3"
        final_path = os.path.join(OUTPUT_DIR, final_filename)

        logger.info(f"Merging {len(temp_clips)} clips into {final_filename}...")
        merge_audio_clips(temp_clips, final_path, speakers=clip_speakers)

        for clip in temp_clips:
            try:
                os.remove(clip)
            except Exception as e:
                logger.warning(f"Failed to remove temporary clip {clip}: {e}")

        audio_url = f"/outputs/{final_filename}"
        from parser import get_detected_speakers_with_genders
        speaker_genders = get_detected_speakers_with_genders(parsed_blocks)
        return jsonify({
            "status": "success",
            "audio_url": audio_url,
            "blocks": parsed_blocks,
            "speakers": speakers,
            "speaker_genders": speaker_genders,
        })

    except Exception as e:
        logger.error(f"Error during narration process: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — File Upload (long text / novel)
# ---------------------------------------------------------------------------

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.txt', '.pdf', '.epub']:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # Parse optional voice_map from form data
    voice_map = None
    voice_map_raw = request.form.get('voice_map')
    if voice_map_raw:
        try:
            voice_map = json.loads(voice_map_raw)
        except json.JSONDecodeError:
            logger.warning("Invalid voice_map JSON in upload form data, ignoring.")

    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "job_id": job_id,
        "total_chunks": 0,
        "current_chunk": 0,
        "text_length": 0,
    }

    thread = threading.Thread(target=process_novel, args=(job_id, file_path, voice_map))
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


# ---------------------------------------------------------------------------
# Routes — Job Status
# ---------------------------------------------------------------------------

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host='0.0.0.0', port=port, debug=debug)
