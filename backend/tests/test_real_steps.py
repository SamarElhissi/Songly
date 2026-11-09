from pathlib import Path

import pytest

from songcraft.device import resolve_device
from songcraft.pipeline.analyze import LibrosaBeatAnalyzer
from songcraft.pipeline.beats import LinePlacement
from songcraft.pipeline.factory import build_pipeline
from songcraft.pipeline.fakes import FakeMusicGenerator, write_silent_wav
from songcraft.pipeline.lyrics import Gpt2LyricsGenerator, InstructLyricsGenerator
from songcraft.pipeline.melody import Note
from songcraft.pipeline.mixer import PydubMixer
from songcraft.pipeline.vocals import GttsVocalSynthesizer, MelodicVocalSynthesizer


class _Probe:
    def __init__(self, cuda: bool, mps: bool) -> None:
        self.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: cuda)})()
        self.backends = type("Backends", (), {})()
        self.backends.mps = type("Mps", (), {"is_available": staticmethod(lambda: mps)})()


def test_auto_device_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("songcraft.device._torch", lambda: _Probe(cuda=True, mps=True))
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr("songcraft.device._torch", lambda: _Probe(cuda=False, mps=True))
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr("songcraft.device._torch", lambda: _Probe(cuda=False, mps=False))
    assert resolve_device("auto") == "cpu"


def test_explicit_device_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():
        raise AssertionError("torch should stay unloaded")

    monkeypatch.setattr("songcraft.device._torch", boom)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_real_pipeline_builds_without_loading_weights(settings_factory, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("weights loaded at startup")

    monkeypatch.setattr("songcraft.pipeline.lyrics._load_text_pipeline", boom)
    monkeypatch.setattr("songcraft.pipeline.lyrics._load_instruct", boom)
    settings = settings_factory(
        music_generator="fake",
        lyrics_generator="gpt2",
        vocal_synthesizer="gtts",
        mixer="pydub",
        beat_analyzer="librosa",
    )
    pipeline = build_pipeline(settings)
    assert isinstance(pipeline.music, FakeMusicGenerator)
    assert isinstance(pipeline.lyrics, Gpt2LyricsGenerator)
    assert pipeline.lyrics._pipeline is None

    instructed = build_pipeline(
        settings_factory(
            music_generator="fake",
            lyrics_generator="instruct",
            vocal_synthesizer="melody",
            lyrics_model="Qwen/Qwen2.5-1.5B-Instruct",
        )
    )
    assert isinstance(instructed.lyrics, InstructLyricsGenerator)
    assert instructed.lyrics._model is None
    assert isinstance(instructed.vocals, MelodicVocalSynthesizer)


def test_gpt2_reuses_one_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = {"count": 0}

    class Pipe:
        class tokenizer:
            eos_token_id = 0

        def __call__(self, prompt: str, **kwargs):
            assert kwargs["max_new_tokens"] == 120
            assert kwargs["do_sample"] is True
            assert kwargs["pad_token_id"] == 0
            return [{"generated_text": f"  {prompt}. la la la  "}]

    def load(model_name: str, device: str) -> Pipe:
        loads["count"] += 1
        assert model_name == "gpt2"
        assert device == "cpu"
        return Pipe()

    monkeypatch.setattr("songcraft.pipeline.lyrics._load_text_pipeline", load)
    generator = Gpt2LyricsGenerator("gpt2", "cpu")
    assert generator.generate("rain") == "rain. la la la"
    generator.generate("rain")
    assert loads["count"] == 1


def test_instruct_lyrics_keep_the_requested_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    loads = {"count": 0}
    seen: dict[str, object] = {}

    class Tokenizer:
        eos_token_id = 7

        def apply_chat_template(self, messages, return_tensors="pt", add_generation_prompt=True):
            seen["messages"] = messages
            return torch.tensor([[1, 2, 3]])

        def decode(self, tokens, skip_special_tokens=True) -> str:
            return "Verse 1:\nRain on the window\n2. I wait for you\nStay a while"

    class Model:
        device = "cpu"

        def generate(self, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 48
            extra = torch.ones(1, 4, dtype=input_ids.dtype)
            return torch.cat([input_ids, extra], dim=1)

    def load(model_name: str, device: str):
        loads["count"] += 1
        assert model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert device == "cpu"
        return Model(), Tokenizer()

    monkeypatch.setattr("songcraft.pipeline.lyrics._load_instruct", load)
    generator = InstructLyricsGenerator("Qwen/Qwen2.5-1.5B-Instruct", "cpu")
    assert generator.generate("rain", 2, 5) == "Rain on the window\nI wait for you"
    generator.generate("rain", 2, 5)
    assert loads["count"] == 1
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert "rain" in messages[1]["content"]
    assert "2" in messages[1]["content"]
    assert "5" in messages[1]["content"]


def test_gtts_is_asked_to_save_mp3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def save(text: str, output_path: Path) -> None:
        seen["text"] = text
        seen["path"] = output_path
        output_path.write_bytes(b"ID3")

    monkeypatch.setattr("songcraft.pipeline.vocals._save_speech", save)
    destination = tmp_path / "vocals" / "line_0.mp3"
    GttsVocalSynthesizer().synthesize("hello there", [Note(midi=60, duration=0.5)], destination)
    assert seen == {"text": "hello there", "path": destination}
    assert destination.suffix == ".mp3"
    assert destination.read_bytes().startswith(b"ID3")


def test_librosa_silence_has_no_tempo(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    audio = tmp_path / "silent.wav"
    write_silent_wav(audio, 1.0)
    tempo, beats = LibrosaBeatAnalyzer().analyze(audio, 1.0)
    assert tempo is None
    assert beats == []


def test_pydub_mixer_writes_mp3(tmp_path: Path, require_ffmpeg: None) -> None:
    pytest.importorskip("pydub")
    from pydub.generators import Sine

    instrumental = tmp_path / "instrumental.wav"
    vocal = tmp_path / "line_0.wav"
    final = tmp_path / "final.mp3"
    write_silent_wav(instrumental, 1.0)
    Sine(440).to_audio_segment(duration=300).export(vocal, format="wav")
    PydubMixer().mix(
        instrumental,
        [LinePlacement("la", 0.1, 0.4)],
        [vocal],
        final,
    )
    assert final.is_file()
    assert final.stat().st_size > 0
