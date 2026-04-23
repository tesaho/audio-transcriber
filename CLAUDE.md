# Audio Transcriber

Single-file audio transcription pipeline that combines whisper.cpp (speech-to-text), pyannote (speaker diarization), and GPT-4o-mini (speaker identification + text cleanup) to produce speaker-labeled markdown transcripts from podcast episodes or any multi-speaker audio.

## Prerequisites

### Required

```bash
# whisper.cpp — speech-to-text engine
brew install whisper-cpp
whisper-cpp --download-model medium.en   # ~1.5 GB, English-only

# ffmpeg — audio format conversion
brew install ffmpeg

# Python dependencies
pip install -r requirements.txt
```

### Speaker diarization (recommended)

```bash
pip install pyannote.audio torch

# Accept pyannote terms at: https://hf.co/pyannote/speaker-diarization-3.1
# Generate a read token at: https://hf.co/settings/tokens
export HUGGINGFACE_TOKEN=hf_...
```

### GPT-enhanced features (optional)

Speaker name identification and text cleanup use OpenAI GPT-4o-mini. These features degrade gracefully without an API key:
- Without key: speakers labeled "Host" / "Guest 1" / "Guest 2" by speaking time
- Without key: text cleanup is regex-only (fixes punctuation spacing, contractions)

```bash
export OPENAI_API_KEY=sk-...
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `HUGGINGFACE_TOKEN` | For diarization | — | pyannote speaker diarization model access |
| `OPENAI_API_KEY` | No | — | GPT speaker naming + text cleanup |
| `WHISPER_MODEL` | No | `medium.en` | Whisper model name or path to .bin file |
| `WHISPER_CPP_CMD` | No | auto-detect | Full path to whisper-cpp binary |

## Usage

### Basic transcription

```bash
python transcriber.py --audio episode.mp3 --title "My Episode"
```

### With domain vocabulary preset

```bash
python transcriber.py --audio episode.mp3 --vocab presets/crypto.txt
```

### Save to file

```bash
python transcriber.py --audio episode.mp3 --output transcript.md
```

### Check RSS feed for existing transcript first

```bash
python transcriber.py --audio episode.mp3 \
    --feed-url https://feeds.megaphone.fm/mypodcast \
    --episode-title "Guest Name"
```

### Single speaker (skip diarization)

```bash
python transcriber.py --audio lecture.mp3 --no-diarize
```

### No API key mode (regex cleanup only)

```bash
python transcriber.py --audio episode.mp3 --no-cleanup
```

### All CLI flags

| Flag | Description |
|---|---|
| `--audio PATH` | Path to audio file (required) |
| `--title TEXT` | Episode title (default: filename stem) |
| `--context TEXT` | Additional context for speaker ID (episode description, etc.) |
| `--feed-url URL` | RSS feed URL to check for existing transcript |
| `--episode-title TEXT` | Title fragment to match in RSS feed |
| `--vocab PATH` | Path to vocabulary preset file |
| `--output PATH` | Output file (default: stdout) |
| `--model NAME` | Whisper model override (e.g. `large-v3`) |
| `--no-cleanup` | Skip GPT text cleanup |
| `--no-diarize` | Skip speaker diarization |

## Python API

```python
from transcriber import transcribe

# Basic
markdown = transcribe(audio_path="episode.mp3", title="My Episode")

# With all options
markdown = transcribe(
    audio_path="episode.mp3",
    title="My Episode",
    context="Guest: Jane Doe, CEO of ExampleCorp",
    vocab_file="presets/crypto.txt",
    no_diarize=False,
    no_cleanup=False,
)
```

## Architecture

```
Audio file
    │
    ├─► whisper.cpp ──► word-level transcription with timestamps
    │
    ├─► pyannote ──► speaker segments [{speaker, start, end}]
    │
    ▼
_merge_whisper_diarization()  ──► assign speaker to each word, group into turns
    │
    ▼
_fix_turn_boundaries()  ──► correct speaker bleed at turn transitions
    │
    ▼
_identify_speakers()  ──► GPT maps SPEAKER_00 → real names (or fallback)
    │
    ▼
_cleanup_turn()  ──► GPT fixes word fragmentation + adds punctuation (or regex fallback)
    │
    ▼
Speaker-labeled markdown output
```

### Turn boundary correction

Pyannote's speaker change timestamps often lag slightly behind the actual speaker change, causing 1-3 words from the previous speaker to bleed into the next turn. `_fix_turn_boundaries()` detects conversation-starting patterns ("Yeah", "Sure", "I think", etc.) and moves bleed words back to the correct speaker.

### Graceful degradation

| Component | With API key | Without API key |
|---|---|---|
| Transcription | whisper.cpp | whisper.cpp (same) |
| Diarization | pyannote | Skipped (single speaker) |
| Speaker names | GPT-4o-mini | Host / Guest N by speaking time |
| Text cleanup | GPT-4o-mini | Regex: punctuation, contractions, spacing |

## Vocabulary Presets

Domain-specific vocabulary helps GPT correct whisper's misrecognition of specialized terms. Presets are plain text files with one term per line.

### Using a preset

```bash
python transcriber.py --audio episode.mp3 --vocab presets/crypto.txt
```

### Creating a custom preset

Create a text file with domain-specific terms (one per line):

```
# presets/medical.txt
electrocardiogram
angioplasty
myocardial infarction
troponin
```

The terms are injected into the GPT cleanup prompt as vocabulary that whisper commonly misrecognizes.

### Included presets

| File | Domain |
|---|---|
| `presets/crypto.txt` | Crypto, DeFi, blockchain, MEV |

## Testing

```bash
# Unit tests (no external dependencies)
pytest tests/ -v -m "not integration"

# Integration tests (requires whisper-cpp + ffmpeg)
pytest tests/ -v -m "integration"

# All tests
pytest tests/ -v
```

## Output Format

The transcriber produces markdown with speaker labels and timestamps:

```markdown
# Episode Title

**Steve Ehrlich** [00:00]
Welcome to the show. Today we're talking about...

**Josh Lim** [01:23]
Yeah, thanks for having me. I think the market is...
```

## License

CC BY-NC 4.0 — open source for non-commercial use.
