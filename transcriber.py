"""
Audio transcription pipeline with speaker diarization.

  whisper.cpp  — speech-to-text with word-level timestamps
  pyannote     — acoustic speaker diarization
  GPT-4o-mini  — speaker name resolution + text cleanup (optional)

Pipeline:
  1. Check for existing RSS/podcast transcript → return early if found
  2. Run whisper.cpp → word-level transcription
  3. Run pyannote → speaker segments
  4. Merge words with speakers → turn-based transcript
  5. Fix speaker boundary bleed → clean turn transitions
  6. Identify speakers via GPT (or fallback to Host/Guest labels)
  7. Clean up text artifacts via GPT (or fallback to regex-only)
  8. Output speaker-labeled markdown

All OpenAI features degrade gracefully when OPENAI_API_KEY is not set.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import requests

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium.en")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")


# ── Whisper binary discovery ──────────────────────────────────────────────────

def _find_whisper_binary() -> str:
    """Return the whisper-cpp binary path.

    Checks in order:
    1. WHISPER_CPP_CMD env var (explicit override)
    2. 'whisper-cpp' on PATH
    3. Common Homebrew locations on Apple Silicon

    Raises RuntimeError if not found.
    """
    import shutil
    if cmd := os.environ.get("WHISPER_CPP_CMD"):
        return cmd
    if shutil.which("whisper-cpp"):
        return "whisper-cpp"
    for candidate in [
        "/opt/homebrew/bin/whisper-cpp",
        "/opt/homebrew/opt/whisper-cpp/bin/whisper-cpp",
        "/usr/local/bin/whisper-cpp",
    ]:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "whisper-cpp binary not found. Set WHISPER_CPP_CMD to its full path, "
        "e.g.: export WHISPER_CPP_CMD=/opt/homebrew/opt/whisper-cpp/bin/whisper-cpp"
    )


def _resolve_model_path(model: str) -> str:
    """Resolve a model name (e.g. 'medium.en') or path to an absolute path.

    Search order:
      1. If it looks like an existing path, use it as-is.
      2. ~/.cache/whisper-cpp/ggml-{model}.bin
      3. /opt/homebrew/share/whisper-cpp/ggml-{model}.bin
    """
    p = Path(model)
    if p.exists():
        return str(p)
    for candidate in [
        Path.home() / f".cache/whisper-cpp/ggml-{model}.bin",
        Path.home() / f"kb/models/ggml-{model}.bin",
        Path("/opt/homebrew/share/whisper-cpp") / f"ggml-{model}.bin",
    ]:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        f"Whisper model '{model}' not found. "
        f"Run: whisper-cpp --download-model {model}  "
        f"or set WHISPER_MODEL to the full path of the .bin file."
    )


# ── RSS transcript check ─────────────────────────────────────────────────────

def _check_rss_transcript(feed_entry: dict) -> str | None:
    """Return transcript text if RSS entry has a <podcast:transcript> URL."""
    transcript_url = feed_entry.get("podcast_transcript_url")
    if not transcript_url:
        return None
    r = requests.get(transcript_url, timeout=30)
    if r.status_code != 200:
        return None
    if "srt" in transcript_url or "vtt" in transcript_url:
        return _strip_srt_vtt(r.text)
    return r.text


def _strip_srt_vtt(raw: str) -> str:
    """Remove SRT/VTT timestamps and sequence numbers, return plain text."""
    cleaned = []
    for line in raw.splitlines():
        line = line.strip()
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"[\d:,\. ]+-->", line):
            continue
        if line.startswith("WEBVTT"):
            continue
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


# ── Whisper transcription ─────────────────────────────────────────────────────

def _transcribe_whisper(audio_path: Path) -> dict:
    """Run whisper.cpp on audio_path.

    Returns normalized dict: {"segments": [{"words": [{"word", "start", "end"}]}]}
    """
    out_json = audio_path.with_suffix(".json")
    if not out_json.exists():
        subprocess.run(
            [
                _find_whisper_binary(),
                "--model", _resolve_model_path(WHISPER_MODEL),
                "--output-json-full",
                "--output-file", str(audio_path.with_suffix("")),
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    with open(out_json) as f:
        raw = json.load(f)

    # Normalize whisper-cli JSON (transcription[].tokens[]) →
    # {"segments": [{"words": [{"word", "start", "end"}]}]}
    segments = []
    for seg in raw.get("transcription", []):
        words = []
        for tok in seg.get("tokens", []):
            text = tok["text"]
            if text.startswith("[_"):  # skip special tokens
                continue
            words.append({
                "word": text,
                "start": tok["offsets"]["from"] / 1000.0,
                "end": tok["offsets"]["to"] / 1000.0,
            })
        if words:
            segments.append({"words": words})
    return {"segments": segments}


# ── Speaker diarization ───────────────────────────────────────────────────────

def _diarize(audio_path: Path) -> list[dict]:
    """Run pyannote speaker diarization. Returns [{speaker, start, end}, ...]."""
    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipeline.to(device)
    result = pipeline(str(audio_path))
    # pyannote 4.x returns DiarizeOutput; earlier versions return Annotation directly
    annotation = getattr(result, "speaker_diarization", result)
    return [
        {"speaker": speaker, "start": turn.start, "end": turn.end}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


# ── Merge whisper + diarization ───────────────────────────────────────────────

def _merge_whisper_diarization(whisper_result: dict, diarization: list[dict]) -> list[dict]:
    """Assign a speaker to each whisper word, group consecutive same-speaker words into turns."""

    def speaker_at(t: float) -> str:
        for seg in diarization:
            if seg["start"] <= t <= seg["end"]:
                return seg["speaker"]
        if not diarization:
            return "UNKNOWN"
        # Snap to nearest segment — pyannote leaves small confidence gaps
        # between speaker turns; words in those gaps belong to the nearest speaker
        nearest = min(diarization, key=lambda s: min(abs(t - s["start"]), abs(t - s["end"])))
        return nearest["speaker"]

    turns: list[dict] = []
    current_speaker: str | None = None
    current_words: list[dict] = []

    for segment in whisper_result.get("segments", []):
        for word in segment.get("words", []):
            mid = (word["start"] + word["end"]) / 2
            spk = speaker_at(mid)
            if spk != current_speaker:
                if current_words:
                    turns.append({
                        "speaker": current_speaker,
                        "start": current_words[0]["start"],
                        "end": current_words[-1]["end"],
                        "text": " ".join(w["word"].strip() for w in current_words),
                    })
                current_speaker = spk
                current_words = [word]
            else:
                current_words.append(word)

    if current_words:
        turns.append({
            "speaker": current_speaker,
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "text": " ".join(w["word"].strip() for w in current_words),
        })
    return _fix_turn_boundaries(turns)


# ── Turn boundary correction ─────────────────────────────────────────────────

# Conversation-starting words/phrases that signal the real start of a new speaker's turn
_TURN_STARTERS = re.compile(
    r"^(yeah|yes|sure|absolutely|right|so|well|no|okay|ok|exactly|"
    r"definitely|totally|honestly|actually|i think|i mean|i would|"
    r"i do|i don't|that's right|that's correct|thanks|thank you|"
    r"and i|but i)\b",
    re.IGNORECASE,
)


def _fix_turn_boundaries(turns: list[dict]) -> list[dict]:
    """Fix speaker bleed at turn boundaries.

    Pyannote diarization timestamps often lag slightly behind the actual
    speaker change, causing the last 1-3 words of speaker A to appear at
    the start of speaker B's turn.  This function detects the pattern
    (a short sentence fragment followed by a conversation-starting word)
    and moves the fragment back to the previous speaker.
    """
    if len(turns) < 2:
        return turns

    for i in range(1, len(turns)):
        prev = turns[i - 1]
        curr = turns[i]

        # Only fix across different speakers
        if prev["speaker"] == curr["speaker"]:
            continue

        text = curr["text"]
        words = text.split()
        if len(words) < 3:
            continue

        # Look for a turn-starter within the first 6 words
        best_split = None
        for j in range(1, min(6, len(words))):
            remainder = " ".join(words[j:])
            if _TURN_STARTERS.match(remainder):
                best_split = j
                break

        if best_split is None:
            continue

        # Move the leading fragment to the previous turn
        fragment = " ".join(words[:best_split])
        prev["text"] = prev["text"] + " " + fragment
        curr["text"] = " ".join(words[best_split:])

    # Second pass: move trailing turn-starters from previous speaker to next.
    # Handles: prev ends with "...thoughts on that. Yeah," and next starts
    # with "Absolutely." — the "Yeah," belongs to the next speaker.
    for i in range(len(turns) - 1, 0, -1):
        prev = turns[i - 1]
        curr = turns[i]

        if prev["speaker"] == curr["speaker"]:
            continue

        prev_words = prev["text"].split()
        if len(prev_words) < 3:
            continue

        # Check if last 1-2 words of prev are a turn-starter
        for trim in (1, 2):
            candidate = " ".join(prev_words[-trim:])
            if _TURN_STARTERS.match(candidate):
                prev["text"] = " ".join(prev_words[:-trim])
                curr["text"] = candidate + " " + curr["text"]
                break

    return turns


# ── Speaker identification ────────────────────────────────────────────────────

def _identify_speakers(client, turns: list[dict], context: str = "") -> dict:
    """Map SPEAKER_00/01/... → real names via GPT-4o-mini. Returns {} on failure."""
    # Use more turns and longer excerpts for better identification
    sample = "\n".join(
        f"[{t['speaker']}]: {t['text'][:400]}" for t in turns[:80]
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at identifying speakers in podcast transcripts. "
                    "You will be given the opening of a transcript with temporary labels "
                    "(SPEAKER_00, SPEAKER_01, etc.) and context about the episode.\n\n"
                    "Identification strategies:\n"
                    "- Hosts typically introduce themselves early: 'My name is ...', "
                    "'I'm ..., your host'\n"
                    "- Hosts introduce guests: 'I'm delighted to bring in ...', "
                    "'joining me today is ...'\n"
                    "- Speakers reading ads/sponsors are usually a separate voice-over "
                    "or the host — label them 'Ad Read' if they only appear in ad segments\n"
                    "- The episode title and description often contain the guest's name and title\n"
                    "- Note: whisper transcription may fragment names (e.g. 'Steve E hr lich' "
                    "= 'Steve Ehrlich', 'Falcon X' = 'FalconX'). Mentally reconstruct "
                    "fragmented names.\n\n"
                    'Return JSON: {"SPEAKER_00": "Real Name", "SPEAKER_01": "Real Name"}. '
                    "Use null ONLY if you truly cannot determine the name from any available "
                    "evidence. Prefer a best guess over null."
                ),
            },
            {"role": "user", "content": f"Episode context: {context}\n\n{sample}"},
        ],
    )
    try:
        raw = resp.choices[0].message.content.strip()
        # GPT sometimes wraps JSON in markdown code blocks
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return {}


def _fallback_speaker_names(turns: list[dict]) -> dict:
    """Assign Host/Guest labels based on cumulative speaking time.

    Called when OPENAI_API_KEY is absent or GPT speaker naming fails.
    Speaker with the most total speaking time becomes 'Host'.
    Others become 'Guest 1', 'Guest 2', etc. in descending time order.
    """
    time_by_speaker: dict[str, float] = {}
    for turn in turns:
        spk = turn["speaker"]
        duration = max(0.0, turn["end"] - turn["start"])
        time_by_speaker[spk] = time_by_speaker.get(spk, 0.0) + duration
    ranked = sorted(time_by_speaker.items(), key=lambda x: x[1], reverse=True)
    name_map: dict[str, str] = {}
    for i, (spk, _) in enumerate(ranked):
        name_map[spk] = "Host" if i == 0 else f"Guest {i}"
    return name_map


# ── Text cleanup ──────────────────────────────────────────────────────────────

def _fix_whisper_spacing(text: str) -> str:
    """Fix common whisper spacing artifacts without an API call.

    Handles: extra spaces around punctuation (" , " → ", "),
    contractions (" 's " → "'s"), and leading punctuation (". Hello" → "Hello").
    """
    # Fix spaces before punctuation: " , " → ", "  " . " → ". "
    text = re.sub(r"\s+([,\.!\?;:])", r"\1", text)
    # Fix spaces around contractions: " 's " → "'s ", " n't " → "n't "
    text = re.sub(r"\s+'(s|t|re|ve|ll|d|m)\b", r"'\1", text)
    # Fix " n't" pattern
    text = re.sub(r"\sn't\b", "n't", text)
    # Fix "I 'm" → "I'm"
    text = re.sub(r"\b(I|you|we|they|he|she|it)\s+'(m|re|ve|ll|d)\b",
                  lambda m: f"{m.group(1)}'{m.group(2)}", text, flags=re.IGNORECASE)
    # Strip leading punctuation (". Hello" from bleed)
    text = re.sub(r"^[,\.!\?;:\s]+", "", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text).strip()
    return text


def _load_vocab(vocab_file: str | Path | None) -> list[str]:
    """Load domain vocabulary from a text file (one term per line)."""
    if not vocab_file:
        return []
    p = Path(vocab_file)
    if not p.exists():
        print(f"  [warn] Vocabulary file not found: {p}")
        return []
    terms = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return terms


def _cleanup_turn(client, text: str, vocab_terms: list[str] | None = None) -> str:
    """Run a GPT-4o-mini pass over a turn to fix whisper artifacts.

    Fixes word-boundary fragmentation, punctuation, and adds paragraph breaks.
    Returns the corrected text, or the original text on any error.
    """
    vocab_section = ""
    if vocab_terms:
        terms_str = ", ".join(vocab_terms)
        vocab_section = (
            f"2. DOMAIN VOCABULARY — these terms are commonly misrecognized by "
            f"whisper. Correct to the right spelling:\n"
            f"   {terms_str}\n\n"
        )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a transcript editor fixing whisper speech-to-text artifacts. "
                        "Apply these corrections:\n\n"
                        "1. WORD BOUNDARY FRAGMENTATION — whisper splits words with spaces.\n"
                        "   Reconstruct fragmented words by joining split syllables.\n\n"
                        f"{vocab_section}"
                        f"{'3' if vocab_section else '2'}. PUNCTUATION — add sentence boundaries "
                        "where missing. Fix run-on sentences. Preserve meaning exactly.\n\n"
                        f"{'4' if vocab_section else '3'}. PARAGRAPH BREAKS — add a blank line "
                        "on major topic shifts.\n\n"
                        f"{'5' if vocab_section else '4'}. DO NOT change the speaker's meaning, "
                        "add words, remove content, or summarize. Fix only transcription "
                        "artifacts.\n\n"
                        "Return only the corrected text, no commentary."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text


# ── Main pipeline ─────────────────────────────────────────────────────────────

def transcribe(
    audio_path: Path,
    feed_entry: dict | None = None,
    context: str = "",
    title: str = "Transcript",
    vocab_file: str | Path | None = None,
    no_diarize: bool = False,
    no_cleanup: bool = False,
) -> str:
    """Full transcription pipeline. Returns turn-based markdown.

    Args:
        audio_path: Path to audio file (mp3, wav, m4a, etc.)
        feed_entry: RSS feed entry dict (checks for podcast:transcript tag)
        context: Additional context for speaker identification (episode description, etc.)
        title: Episode title (used in markdown header and speaker identification)
        vocab_file: Path to vocabulary preset file for domain-specific cleanup
        no_diarize: Skip speaker diarization (single-speaker output)
        no_cleanup: Skip GPT text cleanup (regex-only)
    """
    audio_path = Path(audio_path)

    if feed_entry:
        rss_text = _check_rss_transcript(feed_entry)
        if rss_text:
            print(f"  Using RSS transcript for {title}")
            return rss_text

    print(f"  Running whisper.cpp ({WHISPER_MODEL}) on {audio_path.name}...")
    whisper_result = _transcribe_whisper(audio_path)

    if no_diarize:
        diarization = []
    else:
        if not HF_TOKEN:
            print("  [warn] HUGGINGFACE_TOKEN not set — skipping speaker diarization")
            diarization = []
        else:
            print("  Running pyannote speaker diarization...")
            diarization = _diarize(audio_path)

    turns = _merge_whisper_diarization(whisper_result, diarization)

    # Build rich context for speaker identification
    speaker_context = f"Episode title: {title}"
    if context:
        speaker_context += f"\n{context}"
    if feed_entry:
        desc = feed_entry.get("summary", feed_entry.get("description", ""))
        if desc:
            speaker_context += f"\nEpisode description: {desc[:500]}"
        feed_title = feed_entry.get("feed_title", "")
        if feed_title:
            speaker_context += f"\nPodcast: {feed_title}"

    # Speaker identification (requires OpenAI)
    name_map = {}
    try:
        from openai import OpenAI
        client = OpenAI()
        name_map = _identify_speakers(client, turns, speaker_context)
    except Exception as e:
        print(f"  [warn] Speaker naming skipped: {e}")
    if not name_map:
        name_map = _fallback_speaker_names(turns)
    for turn in turns:
        turn["speaker"] = name_map.get(turn["speaker"]) or turn["speaker"]

    # Text cleanup
    _openai_client = None
    if not no_cleanup:
        try:
            from openai import OpenAI as _OAI
            _openai_client = _OAI()
        except Exception:
            pass

    vocab_terms = _load_vocab(vocab_file)

    lines = [f"# {title}\n"]
    for turn in turns:
        mins, secs = divmod(int(turn["start"]), 60)
        lines.append(f"**{turn['speaker']}** [{mins:02d}:{secs:02d}]")
        turn_text = turn["text"]
        if len(turn_text.split()) > 50 and _openai_client is not None:
            turn_text = _cleanup_turn(_openai_client, turn_text, vocab_terms)
        else:
            turn_text = _fix_whisper_spacing(turn_text)
        lines.append(turn_text)
        lines.append("")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python transcriber.py --audio episode.mp3 --title "My Episode"

  # With crypto vocabulary preset
  python transcriber.py --audio episode.mp3 --vocab presets/crypto.txt

  # Save to file
  python transcriber.py --audio episode.mp3 --output transcript.md

  # Check RSS feed for existing transcript first
  python transcriber.py --audio episode.mp3 \\
      --feed-url https://feeds.megaphone.fm/mypodcast \\
      --episode-title "Guest Name"

  # Skip diarization (single speaker)
  python transcriber.py --audio lecture.mp3 --no-diarize

  # Skip GPT cleanup (regex-only, no API key needed)
  python transcriber.py --audio episode.mp3 --no-cleanup
""",
    )
    parser.add_argument("--audio", type=Path, required=True,
                        help="Path to audio file (mp3, wav, m4a, etc.)")
    parser.add_argument("--title", default=None,
                        help="Episode title (default: filename stem)")
    parser.add_argument("--context", default="",
                        help="Additional context for speaker identification")
    parser.add_argument("--feed-url", default=None,
                        help="RSS feed URL to check for existing transcript")
    parser.add_argument("--episode-title", default="",
                        help="Title fragment to match in RSS feed")
    parser.add_argument("--vocab", type=Path, default=None,
                        help="Path to vocabulary preset file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path (default: stdout)")
    parser.add_argument("--model", default=None,
                        help="Whisper model override (e.g. large-v3)")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Skip GPT text cleanup (regex-only)")
    parser.add_argument("--no-diarize", action="store_true",
                        help="Skip speaker diarization")
    args = parser.parse_args()

    if args.model:
        global WHISPER_MODEL
        WHISPER_MODEL = args.model

    title = args.title or args.audio.stem

    # Build feed_entry if --feed-url is provided
    feed_entry = None
    if args.feed_url:
        import feedparser
        print(f"  Checking RSS feed for transcript: {args.feed_url}")
        feed = feedparser.parse(args.feed_url)
        entries = feed.entries
        if args.episode_title:
            entries = [e for e in entries
                       if args.episode_title.lower() in e.get("title", "").lower()]
        if entries:
            entry = entries[0]
            transcript_url = entry.get("podcast_transcript", {}).get("url") or \
                             next((l.get("href") for l in entry.get("links", [])
                                   if "transcript" in l.get("rel", "").lower()), None)
            if transcript_url:
                feed_entry = {"podcast_transcript_url": transcript_url}
                print(f"  Found RSS transcript: {transcript_url}")

    result = transcribe(
        audio_path=args.audio,
        feed_entry=feed_entry,
        context=args.context,
        title=title,
        vocab_file=args.vocab,
        no_diarize=args.no_diarize,
        no_cleanup=args.no_cleanup,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print(f"\nTranscript saved to: {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
