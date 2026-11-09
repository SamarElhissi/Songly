from pathlib import Path

from songcraft.pipeline.melody import Note, words_of

_SAMPLE_RATE = 22050


class MelodicVocalSynthesizer:
    """Speak each word, then tune it to the note for that word.

    DiffSinger lives under legacy/ and is not used here. The note list is still
    what a singer needs, and this step follows it.
    """

    def synthesize(self, line: str, notes: list[Note], output_path: Path) -> None:
        import numpy as np

        words = words_of(line)
        paired = _pair(words, notes)
        pieces = []
        for index, (word, note) in enumerate(paired):
            temp = output_path.with_name(f".{output_path.stem}-{index}.mp3")
            try:
                _speak(word, temp)
                pieces.append(_fit_to_note(temp, note))
            finally:
                temp.unlink(missing_ok=True)
        audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
        _write_mp3(_edge_fade(audio, _SAMPLE_RATE), _SAMPLE_RATE, output_path)


class GttsVocalSynthesizer:
    def synthesize(self, line: str, notes: list[Note], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # gTTS returns MP3 bytes and cannot follow pitch. The caller passes an .mp3 path.
        _save_speech(line, output_path)


def _pair(words: list[str], notes: list[Note]) -> list[tuple[str, Note]]:
    if notes and len(notes) == len(words):
        return list(zip(words, notes))
    total = sum(note.duration for note in notes) if notes else 0.35 * len(words)
    slot = total / len(words)
    pitches = [note.midi for note in notes] or [60]
    return [
        (word, Note(midi=pitches[index % len(pitches)], duration=slot))
        for index, word in enumerate(words)
    ]


def _speak(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_speech(text, output_path)


def _save_speech(text: str, output_path: Path) -> None:
    from gtts import gTTS

    gTTS(text=text, lang="en").save(str(output_path))


def _fit_to_note(path: Path, note: Note):
    import numpy as np
    import librosa

    audio, sample_rate = librosa.load(path, sr=_SAMPLE_RATE, mono=True)
    target = max(1, int(_SAMPLE_RATE * note.duration))
    if audio.size == 0:
        return np.zeros(target, dtype=np.float32)
    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)))
    shifted = librosa.effects.pitch_shift(
        audio,
        sr=sample_rate,
        n_steps=float(note.midi - _median_midi(audio, sample_rate)),
    )
    current = max(len(shifted) / sample_rate, 0.05)
    rate = min(4.0, max(0.25, current / max(note.duration, 0.05)))
    try:
        stretched = librosa.effects.time_stretch(shifted, rate=rate)
    except Exception:
        stretched = shifted
    if len(stretched) > target:
        stretched = stretched[:target]
    elif len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)))
    return stretched.astype(np.float32)


def _median_midi(audio, sample_rate: int) -> int:
    import numpy as np
    import librosa

    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sample_rate,
    )
    voiced = f0[np.isfinite(f0)] if f0 is not None else np.array([])
    if voiced.size == 0:
        return 57
    return int(round(float(librosa.hz_to_midi(float(np.median(voiced))))))


def _edge_fade(samples, sample_rate: int, milliseconds: int = 15):
    import numpy as np

    fade = int(sample_rate * milliseconds / 1000)
    if fade <= 0 or len(samples) < fade * 2:
        return samples
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    faded = np.array(samples, dtype=np.float32, copy=True)
    faded[:fade] *= ramp
    faded[-fade:] *= ramp[::-1]
    return faded


def _write_mp3(samples, sample_rate: int, output_path: Path) -> None:
    import numpy as np
    from pydub import AudioSegment

    from songcraft.pipeline.mixer import _ensure_ffmpeg

    _ensure_ffmpeg()
    pcm = np.clip(samples, -1.0, 1.0)
    ints = (pcm * 32767).astype(np.int16)
    segment = AudioSegment(
        data=ints.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment.export(output_path, format="mp3")
