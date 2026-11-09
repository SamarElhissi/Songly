import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic_settings import SettingsConfigDict

from songcraft.config import Settings
from songcraft.main import create_app


@pytest.fixture
def require_ffmpeg() -> None:
    """pydub mix/export needs ffmpeg on PATH (or the optional repo ffmpeg/ folder)."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    bundled = Path(__file__).resolve().parents[2] / "ffmpeg"
    has_ffmpeg = (bundled / "ffmpeg.exe").is_file() or (bundled / "ffmpeg").is_file()
    has_ffprobe = (bundled / "ffprobe.exe").is_file() or (bundled / "ffprobe").is_file()
    if has_ffmpeg and has_ffprobe:
        return
    pytest.skip("ffmpeg/ffprobe not available")


class IsolatedSettings(Settings):
    """Ignores the developer environment so tests cannot touch a real database."""

    model_config = SettingsConfigDict(
        env_prefix="SONGCRAFT_TEST_",
        env_file=None,
        extra="ignore",
    )


def make_settings(tmp_path: Path, **overrides: object) -> IsolatedSettings:
    values: dict[str, object] = {
        "database_url": "sqlite:///" + (tmp_path / "test.db").as_posix(),
        "data_dir": tmp_path / "data",
        "device": "cpu",
        "min_duration_seconds": 5,
        "max_duration_seconds": 30,
        "cors_origins": "http://localhost:5173",
        "music_generator": "fake",
        "lyrics_generator": "fake",
        "vocal_synthesizer": "fake",
        "mixer": "fake",
        "beat_analyzer": "fake",
        "job_queue": "inline",
    }
    values.update(overrides)
    return IsolatedSettings(**values)  # type: ignore[arg-type]


@pytest.fixture
def settings(tmp_path: Path) -> IsolatedSettings:
    return make_settings(tmp_path)


@pytest.fixture
def settings_factory(tmp_path: Path):
    def factory(**overrides: object) -> IsolatedSettings:
        return make_settings(tmp_path, **overrides)

    return factory


@pytest.fixture
def client(settings: IsolatedSettings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
