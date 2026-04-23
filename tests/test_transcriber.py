"""
Transcriber unit and integration tests.

Unit tests run with no model dependencies:
    pytest tests/ -v -m "not integration"

Integration tests require whisper-cpp in PATH:
    pytest tests/ -v -m "integration"
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


# ── _strip_srt_vtt ─────────────────────────────────────────────────────────────

def test_strip_srt_removes_timestamps():
    from transcriber import _strip_srt_vtt
    raw = "1\n00:00:01,000 --> 00:00:04,000\nHello world\n\n2\n00:00:05,000 --> 00:00:08,000\nSecond line"
    result = _strip_srt_vtt(raw)
    assert "Hello world" in result
    assert "Second line" in result
    assert "-->" not in result
    assert "00:00:01" not in result


def test_strip_vtt_removes_webvtt_header():
    from transcriber import _strip_srt_vtt
    raw = "WEBVTT\n\n00:00:00.000 --> 00:00:03.000\nHello from VTT"
    result = _strip_srt_vtt(raw)
    assert "Hello from VTT" in result
    assert "WEBVTT" not in result


# ── _check_rss_transcript ──────────────────────────────────────────────────────

def test_check_rss_transcript_none_without_url():
    from transcriber import _check_rss_transcript
    assert _check_rss_transcript({}) is None


def test_check_rss_transcript_none_on_404():
    from transcriber import _check_rss_transcript
    with patch("transcriber.requests.get") as mock:
        mock.return_value = MagicMock(status_code=404)
        result = _check_rss_transcript({"podcast_transcript_url": "https://example.com/t.txt"})
    assert result is None


def test_check_rss_transcript_returns_plain_text():
    from transcriber import _check_rss_transcript
    with patch("transcriber.requests.get") as mock:
        mock.return_value = MagicMock(
            status_code=200,
            text="Transcript text here.",
            headers={"content-type": "text/plain"},
        )
        result = _check_rss_transcript({"podcast_transcript_url": "https://example.com/t.txt"})
    assert result == "Transcript text here."


def test_check_rss_transcript_strips_srt_by_extension():
    from transcriber import _check_rss_transcript
    with patch("transcriber.requests.get") as mock:
        mock.return_value = MagicMock(
            status_code=200,
            text="1\n00:00:01,000 --> 00:00:02,000\nHello world",
            headers={"content-type": "text/plain"},
        )
        result = _check_rss_transcript({"podcast_transcript_url": "https://example.com/t.srt"})
    assert "Hello world" in result
    assert "-->" not in result


# ── _merge_whisper_diarization ─────────────────────────────────────────────────

def test_merge_assigns_speakers_to_words():
    from transcriber import _merge_whisper_diarization
    whisper = {"segments": [{"words": [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "Goodbye", "start": 5.0, "end": 5.5},
    ]}]}
    diarization = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
        {"speaker": "SPEAKER_01", "start": 4.0, "end": 6.0},
    ]
    turns = _merge_whisper_diarization(whisper, diarization)
    assert len(turns) == 2
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert "Hello" in turns[0]["text"]
    assert turns[1]["speaker"] == "SPEAKER_01"
    assert "Goodbye" in turns[1]["text"]


def test_merge_snaps_gap_words_to_nearest_speaker():
    """Words in diarization gaps snap to nearest speaker, not UNKNOWN."""
    from transcriber import _merge_whisper_diarization
    whisper = {"segments": [{"words": [
        {"word": "Hello",   "start": 0.5, "end": 1.0},
        {"word": "name",    "start": 2.0, "end": 2.2},
        {"word": "Goodbye", "start": 4.5, "end": 5.0},
    ]}]}
    diarization = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
        {"speaker": "SPEAKER_01", "start": 4.0, "end": 6.0},
    ]
    turns = _merge_whisper_diarization(whisper, diarization)
    speakers = [t["speaker"] for t in turns]
    assert "UNKNOWN" not in speakers
    assert turns[0]["speaker"] == "SPEAKER_00"
    assert "name" in turns[0]["text"]


def test_merge_unknown_for_unmatched_words():
    from transcriber import _merge_whisper_diarization
    whisper = {"segments": [{"words": [{"word": "Test", "start": 10.0, "end": 10.5}]}]}
    turns = _merge_whisper_diarization(whisper, [])
    assert turns[0]["speaker"] == "UNKNOWN"


def test_merge_empty_returns_empty():
    from transcriber import _merge_whisper_diarization
    assert _merge_whisper_diarization({"segments": []}, []) == []


def test_merge_turn_timestamps():
    from transcriber import _merge_whisper_diarization
    whisper = {"segments": [{"words": [
        {"word": "First", "start": 1.0, "end": 1.5},
        {"word": "second", "start": 1.5, "end": 2.0},
    ]}]}
    turns = _merge_whisper_diarization(whisper, [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}])
    assert turns[0]["start"] == 1.0
    assert turns[0]["end"] == 2.0


# ── _fix_whisper_spacing ──────────────────────────────────────────────────────

def test_fix_whisper_spacing_punctuation():
    from transcriber import _fix_whisper_spacing
    assert _fix_whisper_spacing("Hello , world .") == "Hello, world."


def test_fix_whisper_spacing_contractions():
    from transcriber import _fix_whisper_spacing
    assert _fix_whisper_spacing("I 'm going , you 're right .") == "I'm going, you're right."


def test_fix_whisper_spacing_strips_leading_punctuation():
    from transcriber import _fix_whisper_spacing
    assert _fix_whisper_spacing(". I was just going to add on") == "I was just going to add on"


# ── _fix_turn_boundaries ──────────────────────────────────────────────────────

def test_fix_boundaries_moves_bleed_to_previous_speaker():
    """Words that complete previous speaker's sentence move back."""
    from transcriber import _fix_turn_boundaries
    turns = [
        {"speaker": "A", "text": "What do you see on your", "start": 0, "end": 10},
        {"speaker": "B", "text": "desk? Yeah, sure. I do think ETH", "start": 10, "end": 20},
    ]
    fixed = _fix_turn_boundaries(turns)
    assert fixed[0]["text"].endswith("your desk?")
    assert fixed[1]["text"].startswith("Yeah,")


def test_fix_boundaries_moves_trailing_starter_to_next():
    """Turn-starter at end of previous speaker moves to next speaker."""
    from transcriber import _fix_turn_boundaries
    turns = [
        {"speaker": "A", "text": "your thoughts on that. Yeah,", "start": 0, "end": 10},
        {"speaker": "B", "text": "Absolutely. And like all of us", "start": 10, "end": 20},
    ]
    fixed = _fix_turn_boundaries(turns)
    assert fixed[0]["text"].endswith("on that.")
    assert fixed[1]["text"].startswith("Yeah,")


def test_fix_boundaries_no_change_for_clean_transition():
    from transcriber import _fix_turn_boundaries
    turns = [
        {"speaker": "A", "text": "What are you seeing right now?", "start": 0, "end": 10},
        {"speaker": "B", "text": "Yeah, I think you are right.", "start": 10, "end": 20},
    ]
    fixed = _fix_turn_boundaries(turns)
    assert fixed[0]["text"] == "What are you seeing right now?"
    assert fixed[1]["text"] == "Yeah, I think you are right."


def test_fix_boundaries_same_speaker_unchanged():
    from transcriber import _fix_turn_boundaries
    turns = [
        {"speaker": "A", "text": "first part.", "start": 0, "end": 5},
        {"speaker": "A", "text": "Yeah, same speaker.", "start": 5, "end": 10},
    ]
    fixed = _fix_turn_boundaries(turns)
    assert fixed[0]["text"] == "first part."
    assert fixed[1]["text"] == "Yeah, same speaker."


# ── _identify_speakers ─────────────────────────────────────────────────────────

def test_identify_speakers_returns_name_map():
    from transcriber import _identify_speakers
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='{"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}'
        ))]
    )
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "Thanks"},
    ]
    result = _identify_speakers(mock_client, turns)
    assert result == {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}


def test_identify_speakers_handles_bad_json():
    from transcriber import _identify_speakers
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not json"))]
    )
    assert _identify_speakers(mock_client, [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "Hi"}]) == {}


def test_identify_speakers_strips_markdown_code_block():
    """GPT sometimes wraps JSON in ```json ... ``` code blocks."""
    from transcriber import _identify_speakers
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(
            content='```json\n{"SPEAKER_00": "Steve", "SPEAKER_01": "Josh"}\n```'
        ))]
    )
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "Thanks"},
    ]
    result = _identify_speakers(mock_client, turns)
    assert result == {"SPEAKER_00": "Steve", "SPEAKER_01": "Josh"}


# ── _fallback_speaker_names ───────────────────────────────────────────────────

def test_fallback_speaker_names_assigns_host_to_most_time():
    from transcriber import _fallback_speaker_names
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 60.0, "text": "a"},
        {"speaker": "SPEAKER_01", "start": 60.0, "end": 90.0, "text": "b"},
    ]
    name_map = _fallback_speaker_names(turns)
    assert name_map["SPEAKER_00"] == "Host"
    assert name_map["SPEAKER_01"] == "Guest 1"


def test_fallback_speaker_names_three_speakers():
    from transcriber import _fallback_speaker_names
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0, "text": "x"},
        {"speaker": "SPEAKER_01", "start": 10.0, "end": 25.0, "text": "y"},
        {"speaker": "SPEAKER_02", "start": 25.0, "end": 30.0, "text": "z"},
    ]
    name_map = _fallback_speaker_names(turns)
    assert name_map["SPEAKER_01"] == "Host"
    assert name_map["SPEAKER_00"] == "Guest 1"
    assert name_map["SPEAKER_02"] == "Guest 2"


# ── _load_vocab ───────────────────────────────────────────────────────────────

def test_load_vocab_returns_empty_for_none():
    from transcriber import _load_vocab
    assert _load_vocab(None) == []


def test_load_vocab_reads_file(tmp_path):
    from transcriber import _load_vocab
    f = tmp_path / "vocab.txt"
    f.write_text("MEV\nDeFi\nEthereum\n")
    terms = _load_vocab(f)
    assert terms == ["MEV", "DeFi", "Ethereum"]


# ── Integration tests ──────────────────────────────────────────────────────────

def _whisper_available() -> bool:
    import shutil, os
    if os.environ.get("WHISPER_CPP_CMD"):
        return True
    if shutil.which("whisper-cpp"):
        return True
    for candidate in [
        "/opt/homebrew/bin/whisper-cpp",
        "/opt/homebrew/opt/whisper-cpp/bin/whisper-cpp",
        "/usr/local/bin/whisper-cpp",
    ]:
        if Path(candidate).exists():
            return True
    return False


@pytest.fixture(scope="session")
def sample_audio(tmp_path_factory):
    """10-second 16kHz mono sine wave WAV generated by ffmpeg."""
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("audio") / "sample_10s.wav"
    subprocess.run(
        ["ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
         "-ar", "16000", "-ac", "1", str(out), "-y"],
        check=True, capture_output=True,
    )
    return out


@pytest.mark.integration
@pytest.mark.skipif(not _whisper_available(), reason="whisper-cpp not installed")
def test_transcribe_rss_shortcircuit(sample_audio):
    """RSS transcript shortcircuit skips whisper entirely."""
    from transcriber import transcribe
    with patch("transcriber.requests.get") as mock:
        mock.return_value = MagicMock(
            status_code=200, text="RSS transcript text.",
            headers={"content-type": "text/plain"},
        )
        result = transcribe(
            sample_audio,
            feed_entry={"podcast_transcript_url": "https://example.com/t.txt"},
            title="Test",
        )
    assert "RSS transcript text." in result


@pytest.mark.integration
@pytest.mark.skipif(not _whisper_available(), reason="whisper-cpp not installed")
def test_transcribe_full_pipeline_runs(sample_audio):
    """Full pipeline runs end-to-end on a sine wave without raising."""
    from transcriber import transcribe
    with patch("transcriber._diarize", return_value=[]):
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="{}"))]
            )
            result = transcribe(sample_audio, title="Sine Wave Test")
    assert isinstance(result, str)
