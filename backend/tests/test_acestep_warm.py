import sys
import time
from pathlib import Path

from songcraft.db import Database
from songcraft.models import Song, SongStatus
from songcraft.pipeline.ace_song import AceStepGenerator
from songcraft.pipeline.factory import build_pipeline
from songcraft.pipeline.runner import run_song

FAKE_WORKER = Path(__file__).with_name("fake_acestep_worker.py")


def _generator(settings_factory, tmp_path: Path, **overrides: object) -> AceStepGenerator:
    values: dict[str, object] = {
        "music_generator": "acestep",
        "acestep_root": tmp_path / "ace",
        "acestep_idle_seconds": 0,
        "acestep_song_timeout_seconds": 5,
    }
    values.update(overrides)
    settings = settings_factory(**values)
    command = [sys.executable, "-u", str(FAKE_WORKER), "--serve"]
    return AceStepGenerator(settings, command=command)


def test_warm_worker_starts_lazily_and_reuses_the_process(settings_factory, tmp_path: Path) -> None:
    generator = _generator(settings_factory, tmp_path)
    try:
        assert generator.worker_pid() is None
        first = tmp_path / "one.mp3"
        second = tmp_path / "two.mp3"
        generator.generate_song("a gentle song", 30, first, job_id="song-1")
        pid = generator.worker_pid()
        assert pid is not None
        generator.generate_song("a gentle song", 30, second, job_id="song-2")
        assert generator.worker_pid() == pid
        assert first.read_bytes() == b"ID3-fake"
        assert second.read_bytes() == b"ID3-fake"
    finally:
        generator.shutdown()
        assert generator.worker_pid() is None


def test_warm_worker_shuts_down_when_idle(settings_factory, tmp_path: Path) -> None:
    generator = _generator(settings_factory, tmp_path, acestep_idle_seconds=0.4)
    try:
        generator.generate_song("a gentle song", 30, tmp_path / "idle.mp3", job_id="song-1")
        assert generator.worker_pid() is not None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and generator.worker_pid() is not None:
            time.sleep(0.05)
        assert generator.worker_pid() is None
    finally:
        generator.shutdown()


def test_warm_worker_keeps_going_when_stdout_has_plain_text(
    settings_factory, tmp_path: Path
) -> None:
    generator = _generator(settings_factory, tmp_path)
    try:
        output = tmp_path / "noisy.mp3"
        generator.generate_song("noisy", 30, output, job_id="song-1")
        assert output.read_bytes() == b"ID3-fake"
        assert generator.worker_pid() is not None
    finally:
        generator.shutdown()


def test_warm_worker_restarts_after_a_crash(settings_factory, tmp_path: Path) -> None:
    generator = _generator(settings_factory, tmp_path)
    try:
        try:
            generator.generate_song("crash", 30, tmp_path / "crash.mp3", job_id="song-1")
        except Exception as exc:
            assert str(exc) == "ACE-Step worker stopped unexpectedly"
        else:
            raise AssertionError("crash should fail the song")
        assert generator.worker_pid() is None
        output = tmp_path / "after.mp3"
        generator.generate_song("a gentle song", 30, output, job_id="song-2")
        assert output.read_bytes() == b"ID3-fake"
        assert generator.worker_pid() is not None
    finally:
        generator.shutdown()


def test_warm_worker_restarts_after_a_timeout(settings_factory, tmp_path: Path) -> None:
    generator = _generator(settings_factory, tmp_path, acestep_song_timeout_seconds=0.6)
    try:
        try:
            generator.generate_song("hang", 30, tmp_path / "hang.mp3", job_id="song-1")
        except Exception as exc:
            assert str(exc) == "ACE-Step took too long and was stopped"
        else:
            raise AssertionError("hang should fail the song")
        assert generator.worker_pid() is None
        output = tmp_path / "after.mp3"
        generator.generate_song("a gentle song", 30, output, job_id="song-2")
        assert output.read_bytes() == b"ID3-fake"
    finally:
        generator.shutdown()


def test_warm_worker_records_load_time_on_the_first_song_only(settings_factory, tmp_path: Path) -> None:
    settings = settings_factory(
        music_generator="acestep",
        acestep_root=tmp_path / "ace",
        acestep_idle_seconds=0,
        acestep_song_timeout_seconds=5,
    )
    generator = AceStepGenerator(
        settings, command=[sys.executable, "-u", str(FAKE_WORKER), "--serve"]
    )
    database = Database(settings)
    database.create_all(settings)
    pipeline = build_pipeline(settings)
    pipeline.music = generator
    try:
        for song_id in ("song-1", "song-2"):
            with database.session() as session:
                session.add(
                    Song(
                        id=song_id,
                        prompt="a gentle song",
                        duration_seconds=12,
                        status=SongStatus.queued.value,
                    )
                )
                session.commit()
                run_song(song_id, session, pipeline, settings)
                song = session.get(Song, song_id)
            assert song is not None
            assert song.status == SongStatus.completed.value
            assert song.lyrics == "fake line"
            assert "generating_music" in song.step_timings
            if song_id == "song-1":
                assert song.step_timings["loading_models"] >= 0
            else:
                assert "loading_models" not in song.step_timings
    finally:
        generator.shutdown()


def test_oom_fails_the_song_and_the_next_song_runs(settings_factory, tmp_path: Path) -> None:
    settings = settings_factory(
        music_generator="acestep",
        acestep_root=tmp_path / "ace",
        acestep_idle_seconds=0,
        acestep_song_timeout_seconds=5,
    )
    generator = AceStepGenerator(
        settings, command=[sys.executable, "-u", str(FAKE_WORKER), "--serve"]
    )
    database = Database(settings)
    database.create_all(settings)
    pipeline = build_pipeline(settings)
    pipeline.music = generator
    try:
        with database.session() as session:
            session.add(
                Song(id="oom", prompt="oom", duration_seconds=12, status=SongStatus.queued.value)
            )
            session.commit()
            run_song("oom", session, pipeline, settings)
            failed = session.get(Song, "oom")
        assert failed is not None
        assert failed.status == SongStatus.failed.value
        assert failed.error_message == "ACE-Step ran out of GPU memory"
        assert generator.worker_pid() is None
        with database.session() as session:
            session.add(
                Song(
                    id="next",
                    prompt="a gentle song",
                    duration_seconds=12,
                    status=SongStatus.queued.value,
                )
            )
            session.commit()
            run_song("next", session, pipeline, settings)
            song = session.get(Song, "next")
        assert song is not None
        assert song.status == SongStatus.completed.value
    finally:
        generator.shutdown()
