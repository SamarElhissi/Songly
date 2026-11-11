import json
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from songcraft.config import Settings
from songcraft.models import Song, SongStatus
from songcraft.paths import final_path, instrumental_path


class SongCreate(BaseModel):
    prompt: str = Field(min_length=3, max_length=500)
    duration_seconds: float

    @field_validator("prompt", mode="before")
    @classmethod
    def strip_prompt(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("duration_seconds")
    @classmethod
    def finite_duration(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("duration_seconds must be a finite number")
        return value


class TrackFlags(BaseModel):
    instrumental: bool
    final: bool


class SongRead(BaseModel):
    id: str
    prompt: str
    duration_seconds: float
    status: SongStatus
    error_message: str | None
    lyrics: str | None
    tempo_bpm: float | None
    step_timings: dict[str, float]
    # acestep shows one progress step. staged is the older five-step path.
    engine: str
    tracks: TrackFlags
    created_at: datetime
    updated_at: datetime


def song_to_read(song: Song, settings: Settings) -> SongRead:
    return SongRead(
        id=song.id,
        prompt=song.prompt,
        duration_seconds=song.duration_seconds,
        status=SongStatus(song.status),
        error_message=song.error_message,
        lyrics=song.lyrics,
        tempo_bpm=song.tempo_bpm,
        step_timings=_timings(song.step_timings),
        engine="acestep" if settings.music_generator == "acestep" else "staged",
        tracks=TrackFlags(
            instrumental=instrumental_path(settings, song.id).is_file(),
            final=final_path(settings, song.id).is_file(),
        ),
        created_at=_as_utc(song.created_at),
        updated_at=_as_utc(song.updated_at),
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        # SQLite gives back naive datetimes; they were stored as UTC.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timings(value: object) -> dict[str, float]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return {str(key): float(item) for key, item in value.items()}
    return {}
