from collections.abc import Callable

from songcraft.config import Settings
from songcraft.pipeline.ace_song import AceStepGenerator
from songcraft.pipeline.analyze import LibrosaBeatAnalyzer
from songcraft.pipeline.fakes import (
    FakeBeatAnalyzer,
    FakeLyricsGenerator,
    FakeMixer,
    FakeMusicGenerator,
    FakeVocalSynthesizer,
)
from songcraft.pipeline.lyrics import Gpt2LyricsGenerator, InstructLyricsGenerator
from songcraft.pipeline.mixer import PydubMixer
from songcraft.pipeline.vocals import GttsVocalSynthesizer, MelodicVocalSynthesizer
from songcraft.pipeline.interfaces import (
    BeatAnalyzer,
    LyricsGenerator,
    Mixer,
    MusicGenerator,
    Pipeline,
    VocalSynthesizer,
)


def build_pipeline(settings: Settings) -> Pipeline:
    # One instance per process, so wrappers can keep loaded weights here.
    return Pipeline(
        music=_select(
            settings.music_generator,
            {
                "fake": FakeMusicGenerator,
                "acestep": lambda: AceStepGenerator(settings),
            },
            "music generator",
        ),
        lyrics=_select(
            settings.lyrics_generator,
            {
                "fake": FakeLyricsGenerator,
                "instruct": lambda: InstructLyricsGenerator(settings.lyrics_model, settings.device),
                "gpt2": lambda: Gpt2LyricsGenerator(settings.lyrics_model, settings.device),
            },
            "lyrics generator",
        ),
        vocals=_select(
            settings.vocal_synthesizer,
            {
                "fake": FakeVocalSynthesizer,
                "melody": MelodicVocalSynthesizer,
                "gtts": GttsVocalSynthesizer,
            },
            "vocal synthesizer",
        ),
        mixer=_select(
            settings.mixer,
            {"fake": FakeMixer, "pydub": PydubMixer},
            "mixer",
        ),
        beats=_select(
            settings.beat_analyzer,
            {"fake": FakeBeatAnalyzer, "librosa": LibrosaBeatAnalyzer},
            "beat analyzer",
        ),
    )


def _select(
    name: str,
    options: dict[str, Callable[[], MusicGenerator | LyricsGenerator | VocalSynthesizer | Mixer | BeatAnalyzer]],
    label: str,
) -> MusicGenerator | LyricsGenerator | VocalSynthesizer | Mixer | BeatAnalyzer:
    try:
        builder = options[name]
    except KeyError:
        known = ", ".join(sorted(options))
        raise ValueError(f"Unknown {label} '{name}'. Expected one of: {known}.") from None
    return builder()
