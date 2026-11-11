import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session

from songcraft.clock import utcnow
from songcraft.config import Settings
from songcraft.models import STEP_ORDER, Song, SongStatus
from songcraft.paths import final_path, instrumental_path, song_dir, vocal_path
from songcraft.pipeline.ace_song import AceStepGenerator, AceStepWorkerError
from songcraft.pipeline.beats import LinePlacement, map_lines_to_beats
from songcraft.pipeline.interfaces import Pipeline
from songcraft.pipeline.melody import build_line_melody, estimate_key, syllables_per_line, target_line_count
from songcraft.pipeline.text import parse_lyric_lines, split_lyrics_into_lines

logger = logging.getLogger("songcraft.pipeline")

_FAILURE_MESSAGES = {
    SongStatus.generating_music: "Music generation failed",
    SongStatus.analyzing_beats: "Beat analysis failed",
    SongStatus.writing_lyrics: "Lyrics generation failed",
    SongStatus.synthesizing_vocals: "Vocal synthesis failed",
    SongStatus.mixing: "Mixing failed",
}

T = TypeVar("T")


class PipelineError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class SongRemoved(Exception):
    """Raised when the row disappears mid-run, for example after a delete."""


def run_song(song_id: str, session: Session, pipeline: Pipeline, settings: Settings) -> None:
    song = session.get(Song, song_id)
    if song is None or song.status in {SongStatus.completed.value, SongStatus.failed.value}:
        return

    song_dir(settings, song.id).mkdir(parents=True, exist_ok=True)
    instrumental = instrumental_path(settings, song.id)
    final = final_path(settings, song.id)

    try:
        if isinstance(pipeline.music, AceStepGenerator):
            _run_acestep(session, song, pipeline, final)
            return
        _run_step(
            session,
            song,
            SongStatus.generating_music,
            lambda: pipeline.music.generate(song.prompt, song.duration_seconds, instrumental),
        )

        def analyze() -> tuple[float | None, list[float]]:
            tempo, beat_times = pipeline.beats.analyze(instrumental, song.duration_seconds)
            song.tempo_bpm = tempo
            return tempo, beat_times

        _tempo, beat_times = _run_step(session, song, SongStatus.analyzing_beats, analyze)

        def write_lyrics() -> list[LinePlacement]:
            count = target_line_count(song.duration_seconds)
            syllables = syllables_per_line(song.duration_seconds, count)
            text = pipeline.lyrics.generate(song.prompt, count, syllables)
            lines = parse_lyric_lines(text or "")
            # A single paragraph is still the old prose shape, so split that on punctuation.
            if len(lines) <= 1:
                lines = split_lyrics_into_lines(text or "")
            lines = lines[:count]
            song.lyrics = "\n".join(lines)
            return map_lines_to_beats(lines, beat_times, song.duration_seconds)

        placements = _run_step(session, song, SongStatus.writing_lyrics, write_lyrics)

        def synthesize() -> list[Path]:
            key = estimate_key(instrumental)
            paths: list[Path] = []
            for index, placement in enumerate(placements):
                path = vocal_path(settings, song.id, index)
                notes = build_line_melody(placement.line, placement.end - placement.start, key)
                pipeline.vocals.synthesize(placement.line, notes, path)
                paths.append(path)
            return paths

        vocal_paths = _run_step(session, song, SongStatus.synthesizing_vocals, synthesize)
        _run_step(
            session,
            song,
            SongStatus.mixing,
            lambda: pipeline.mixer.mix(instrumental, placements, vocal_paths, final),
        )
        _set_status(session, song, SongStatus.completed)
    except SongRemoved:
        logger.info("song_removed job_id=%s", song_id)
    except PipelineError as exc:
        _mark_failed(session, song, exc.message)
    except Exception:
        logger.exception("song_failed job_id=%s", song_id)
        _mark_failed(session, song, "Song generation failed")


def _run_acestep(session: Session, song: Song, pipeline: Pipeline, final: Path) -> None:
    def generate() -> None:
        def on_loaded(seconds: float) -> None:
            # Commit before generation so a poll can show the load on this song.
            song.step_timings = {**(song.step_timings or {}), "loading_models": round(seconds, 3)}
            flag_modified(song, "step_timings")
            session.add(song)
            session.commit()

        outcome = pipeline.music.generate_song(
            song.prompt,
            song.duration_seconds,
            final,
            job_id=song.id,
            on_loaded=on_loaded,
        )
        song.lyrics = str(outcome.get("lyrics") or "")
        bpm = outcome.get("bpm")
        song.tempo_bpm = float(bpm) if isinstance(bpm, (int, float)) else None
        duration = outcome.get("duration")
        if isinstance(duration, (int, float)):
            song.duration_seconds = float(duration)
        generating = outcome.get("generating_music")
        if isinstance(generating, (int, float)):
            song.step_timings = {**(song.step_timings or {}), "generating_music": round(float(generating), 3)}
        loading = outcome.get("loading_models")
        if isinstance(loading, (int, float)) and "loading_models" not in (song.step_timings or {}):
            song.step_timings = {**(song.step_timings or {}), "loading_models": round(float(loading), 3)}

    _run_step(session, song, SongStatus.generating_music, generate)
    _set_status(session, song, SongStatus.completed)


def _run_step(
    session: Session,
    song: Song,
    step: SongStatus,
    fn: Callable[[], T],
) -> T:
    if step not in STEP_ORDER:
        raise PipelineError("Song generation failed")
    _set_status(session, song, step)
    started = time.perf_counter()
    try:
        result = fn()
    except SongRemoved:
        raise
    except AceStepWorkerError as exc:
        elapsed = time.perf_counter() - started
        logger.info(
            "step_finished job_id=%s step=%s duration_s=%.3f status=failed",
            song.id,
            step.value,
            elapsed,
        )
        raise PipelineError(exc.message) from None
    except Exception:
        elapsed = time.perf_counter() - started
        logger.info(
            "step_finished job_id=%s step=%s duration_s=%.3f status=failed",
            song.id,
            step.value,
            elapsed,
        )
        logger.exception("step_error job_id=%s step=%s", song.id, step.value)
        raise PipelineError(_FAILURE_MESSAGES[step]) from None
    elapsed = time.perf_counter() - started
    timings = {**(song.step_timings or {})}
    if step.value not in timings:
        timings[step.value] = round(elapsed, 3)
    song.step_timings = timings
    flag_modified(song, "step_timings")
    song.updated_at = utcnow()
    session.add(song)
    session.commit()
    logger.info(
        "step_finished job_id=%s step=%s duration_s=%.3f status=ok",
        song.id,
        step.value,
        elapsed,
    )
    return result


def _set_status(session: Session, song: Song, status: SongStatus) -> None:
    # Commit before the slow work so a poll sees the step that is running.
    current = session.get(Song, song.id)
    if current is None:
        raise SongRemoved()
    current.status = status.value
    current.updated_at = utcnow()
    session.add(current)
    session.commit()
    song.status = current.status
    song.updated_at = current.updated_at


def _mark_failed(session: Session, song: Song, message: str) -> None:
    current = session.get(Song, song.id)
    if current is None:
        return
    current.status = SongStatus.failed.value
    current.error_message = message
    current.updated_at = utcnow()
    session.add(current)
    session.commit()
