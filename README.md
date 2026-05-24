# 🎙️ VoiceWeave — Expressive AI Narration Platform

> *Turn plain story text into emotionally expressive multi-speaker audio. Built for comics, novels, and story platforms.*

---

## 🚀 What Is This?

**VoiceWeave** is a hackathon MVP that converts dialogue-heavy story text into expressive, multi-speaker audio narration — automatically detecting emotion, character, and tone.

**Input:**
```
"No... don't leave me," she whispered.
"Enough!" he shouted.
```

**Output:** A rendered audio file with distinct voices, emotional delivery, and pacing.

---

## ⚡ Built With Gemini CLI (Vibe-Coded Fast)

This project is designed to be **vibe-coded quickly using [Gemini CLI](https://github.com/google-gemini/gemini-cli)**. All AI generation logic is handled via Gemini's API, keeping the stack lightweight and fast to ship.

```bash
# Install Gemini CLI
npm install -g @google/gemini-cli

# Authenticate
gemini auth login

# Start vibe-coding
gemini "Build me a Flask API that parses story dialogue, detects speaker emotion, and returns structured JSON"
```

---

## 🖥️ Hardware Requirements (Low-End Friendly)

Designed to run on **budget/older hardware** — tested on equivalent of a 4 GB VRAM laptop:

| Component | Minimum | Notes |
|-----------|---------|-------|
| RAM | 8 GB | 16 GB recommended |
| VRAM | **4 GB** | ✅ Fully supported |
| GPU | Any CUDA/CPU | CPU fallback works |
| Storage | 5 GB free | For model cache |
| OS | Windows / Linux / macOS | All supported |

### ✅ No VRAM Conflicts — Here's How:

- **TTS Model**: Uses [Coqui TTS](https://github.com/coqui-ai/TTS) (`tts_models/en/ljspeech/fast_pitch`) — runs fine on 4 GB VRAM or CPU
- **No LLaMA / No Stable Diffusion** — avoids the heavy models that blow up VRAM
- **Emotion Detection**: Handled via Gemini API (cloud) — zero local VRAM cost
- **Speaker Diarization**: Lightweight rule-based parser, no ML needed

---

## 🏗️ Architecture (MVP)

```
┌─────────────────────────────────────────┐
│              VoiceWeave MVP             │
│                                         │
│  ┌─────────┐    ┌──────────────────┐   │
│  │  Input  │───▶│  Gemini API      │   │
│  │  Text   │    │  (Parse + Emotion│   │
│  └─────────┘    │   Detection)     │   │
│                 └────────┬─────────┘   │
│                          │ JSON        │
│                 ┌────────▼─────────┐   │
│                 │   Coqui TTS      │   │
│                 │  (Local, 4GB ok) │   │
│                 └────────┬─────────┘   │
│                          │ .wav/.mp3   │
│                 ┌────────▼─────────┐   │
│                 │  Audio Output    │   │
│                 └──────────────────┘   │
└─────────────────────────────────────────┘
```

### Two Modes:

**B2B — Hosted API**
- REST endpoint: `POST /api/narrate`
- Input: raw story text + API key
- Output: audio file URL or base64

**B2C — Web App**
- Upload `.txt`, `.pdf`, or paste text
- Click "Generate"
- Download narrated audio

---

## 📦 Stack

| Layer | Tech | Why |
|-------|------|-----|
| Backend | **Python + Flask** | Fast to scaffold, Gemini CLI loves it |
| AI Parsing | **Gemini 1.5 Flash** (via API) | Free tier, fast, great at structured output |
| TTS | **Coqui TTS** | Open source, 4GB VRAM compatible, CPU fallback |
| Frontend | **Vanilla HTML/JS** or React | Keep it simple for hackathon |
| Audio Mix | **pydub** | Merge multi-speaker clips |

---

## 🛠️ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-handle/voiceweave
cd voiceweave
pip install flask coqui-tts pydub google-generativeai python-dotenv
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com)

### 3. Run

```bash
python app.py
# → http://localhost:5000
```

### 4. Test It

```bash
curl -X POST http://localhost:5000/api/narrate \
  -H "Content-Type: application/json" \
  -d '{"text": "\"No... don'\''t leave me,\" she whispered.\n\"Enough!\" he shouted."}'
```

---

## 📁 Project Structure

```
voiceweave/
├── app.py              # Flask app + API routes
├── parser.py           # Gemini-powered dialogue parser
├── tts_engine.py       # Coqui TTS wrapper
├── audio_mixer.py      # pydub multi-speaker merge
├── static/
│   └── index.html      # Simple web UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🤖 Gemini CLI Prompts Used to Build This

```bash
# Generate the parser
gemini "Write a Python function that takes raw story text and uses the Gemini API to return JSON: [{speaker, dialogue, emotion, tone}]"

# Generate TTS wrapper
gemini "Write a Python wrapper for Coqui TTS that maps emotion tags to speaking rate and pitch adjustments, with CPU fallback"

# Generate Flask routes
gemini "Write a Flask REST API with a /api/narrate POST endpoint that ties together a dialogue parser and TTS engine"
```

---

## 🎯 Hackathon Scope (MVP Checklist)

- [x] Parse multi-speaker dialogue from raw text
- [x] Detect emotion per line (whisper, shout, cry, etc.)
- [x] Generate audio per speaker with emotional inflection
- [x] Merge into single narrated audio file
- [x] REST API endpoint for B2B integration
- [x] Simple web UI for B2C demo
- [ ] Voice cloning (post-MVP)
- [ ] Real-time streaming audio (post-MVP)
- [ ] Comic panel image sync (post-MVP)

---

## ⚠️ Known Limitations (MVP)

- Coqui TTS voices are not ultra-realistic — good enough for demo
- Emotion detection accuracy depends on Gemini API response quality
- No authentication on API (add before production)
- Audio generation is synchronous (queue it for scale)

---

## 📄 License

MIT — build freely, ship fast.

---

> *Vibe-coded at [Hackathon Name] in [X] hours using Gemini CLI.*
> *Hardware: Laptop with 4GB VRAM. No excuses.*
