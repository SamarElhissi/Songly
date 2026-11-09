import wave
from pathlib import Path

import pytest

from songcraft.pipeline.melody import (
    MAJOR_SCALE,
    KeyEstimate,
    Note,
    build_line_melody,
    estimate_key,
    syllables_per_line,
    target_line_count,
)
from songcraft.pipeline.text import parse_lyric_lines
from songcraft.pipeline.vocals import MelodicVocalSynthesizer


def test_line_budget_follows_the_song_length() -> None:
    assert target_line_count(5) == 2
    assert target_line_count(30) == 8
    assert syllables_per_line(8, 2) == 8


def test_parse_drops_labels_and_keeps_lines() -> None:
    text = "Verse 1:\nRain on the window\n2. I wait for you\nHere are the lines:\n* Stay with me"
    assert parse_lyric_lines(text) == ["Rain on the window", "I wait for you", "Stay with me"]


def test_melody_has_one_note_per_word_and_resolves() -> None:
    key = KeyEstimate(67, MAJOR_SCALE)
    notes = build_line_melody("rain on the glass", 2.0, key)
    assert len(notes) == 4
    assert abs(sum(note.duration for note in notes) - 2.0) < 1e-6
    assert notes[-1].midi == 67
    assert all(60 <= note.midi <= 84 for note in notes)


def test_empty_line_still_gets_a_note() -> None:
    notes = build_line_melody("...", 1.0, KeyEstimate(60, MAJOR_SCALE))
    assert notes == [Note(midi=60, duration=1.0)]


def test_c_major_chord_estimates_c(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    sample_rate = 22050
    times = numpy.linspace(0, 2, sample_rate * 2, endpoint=False)
    chord = (
        numpy.sin(2 * numpy.pi * 261.63 * times)
        + numpy.sin(2 * numpy.pi * 329.63 * times)
        + numpy.sin(2 * numpy.pi * 392.00 * times)
    )
    audio = tmp_path / "c-major.wav"
    _write_wav(audio, chord / numpy.max(numpy.abs(chord)) * 0.5, sample_rate)
    key = estimate_key(audio)
    assert key.tonic_midi % 12 == 0
    assert key.scale == MAJOR_SCALE


def test_melody_vocal_matches_the_note_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, require_ffmpeg: None
) -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    pytest.importorskip("pydub")

    def speak(text: str, output_path: Path) -> None:
        sample_rate = 22050
        times = numpy.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
        tone = 0.4 * numpy.sin(2 * numpy.pi * 220 * times)
        _write_wav(output_path, tone, sample_rate)

    monkeypatch.setattr("songcraft.pipeline.vocals._speak", speak)
    destination = tmp_path / "vocals" / "line_0.mp3"
    MelodicVocalSynthesizer().synthesize(
        "hello there",
        [Note(midi=64, duration=0.45), Note(midi=67, duration=0.45)],
        destination,
    )
    from pydub import AudioSegment

    clip = AudioSegment.from_file(destination)
    assert 800 < len(clip) < 1100


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    ints = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(ints.tobytes())
