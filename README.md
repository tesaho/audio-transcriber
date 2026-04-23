# Audio Transcriber

Speaker-diarized audio transcription using whisper.cpp + pyannote + GPT-4o-mini.

Produces clean, speaker-labeled markdown transcripts from podcast episodes or any multi-speaker audio. GPT features (speaker naming, text cleanup) are optional — the tool works without an API key.

## Quick Start

```bash
brew install whisper-cpp ffmpeg
whisper-cpp --download-model medium.en
pip install -r requirements.txt

python transcriber.py --audio episode.mp3 --title "My Episode"
```

## Features

- **Speaker diarization** — identifies who said what using pyannote
- **Speaker naming** — GPT resolves "SPEAKER_00" → real names from context
- **Turn boundary correction** — fixes diarization bleed between speakers
- **Text cleanup** — GPT fixes whisper word fragmentation and punctuation
- **Domain vocabulary** — configurable presets for specialized terminology
- **Graceful degradation** — works without any API keys (regex fallback)
- **RSS transcript check** — skips transcription if podcast provides a transcript

## Documentation

See [CLAUDE.md](CLAUDE.md) for full setup instructions, API reference, architecture details, and configuration options.

## License

[CC BY-NC 4.0](LICENSE) — open source for non-commercial use.
