from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SONGCRAFT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite:///./data/songcraft.db"
    data_dir: Path = Path("./data")
    # auto, cuda, mps, or cpu. Passed through to the older lyrics path when used.
    device: str = "auto"
    min_duration_seconds: float = 5
    max_duration_seconds: float = 30
    # Comma-separated so a single environment variable can list origins.
    cors_origins: str = "http://localhost:5173"
    music_generator: str = "acestep"
    # Empty means <repo>/.venv-acestep/Scripts/python.exe. ACE-Step has its own torch.
    acestep_python: str = ""
    acestep_root: Path = Path("./ace-step")
    lyrics_generator: str = "instruct"
    vocal_synthesizer: str = "melody"
    mixer: str = "pydub"
    # Beat tracking is not one of the four model slots; the runner needs times
    # before it can place lyric lines on the older path.
    beat_analyzer: str = "librosa"
    lyrics_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    job_queue: str = "inprocess"

    @model_validator(mode="after")
    def _check_bounds(self) -> "Settings":
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError(
                "max_duration_seconds must be greater than or equal to min_duration_seconds"
            )
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, mps, cpu")
        return self


def clamp_duration(value: float, settings: Settings) -> float:
    return min(settings.max_duration_seconds, max(settings.min_duration_seconds, value))


@lru_cache
def get_settings() -> Settings:
    return Settings()
