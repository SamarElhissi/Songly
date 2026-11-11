import inspect
import json
import subprocess
from pathlib import Path

from songcraft.db import Database
from songcraft.models import Song, SongStatus
from songcraft.paths import final_path, instrumental_path
from songcraft.pipeline.ace_song import AceStepGenerator, acestep_duration
from songcraft.pipeline.acestep_worker import _CallTimer
from songcraft.pipeline.factory import build_pipeline
from songcraft.pipeline.runner import run_song


def test_duration_is_at_least_ten_seconds() -> None:
    assert acestep_duration(8) == 10
    assert acestep_duration(30) == 30
    assert acestep_duration(900) == 600


def test_acestep_builds_without_starting_a_process(settings_factory) -> None:
    pipeline = build_pipeline(settings_factory(music_generator="acestep"))
    assert isinstance(pipeline.music, AceStepGenerator)


def test_acestep_saves_the_mix_and_skips_the_vocal_steps(settings_factory, monkeypatch) -> None:
    settings = settings_factory(music_generator="acestep")
    database = Database(settings)
    database.create_all(settings)
    song_id = "ace-song"
    pipeline = build_pipeline(settings)

    def generate_song(
        self,
        prompt: str,
        duration_seconds: float,
        output_path: Path,
        *,
        job_id: str | None = None,
        on_loaded: object = None,
    ) -> dict:
        assert prompt == "rain"
        assert duration_seconds == 8
        assert job_id == song_id
        output_path.write_bytes(b"ID3-ace")
        return {"lyrics": "Rain on the glass", "bpm": 96, "duration": 10.0}

    monkeypatch.setattr(AceStepGenerator, "generate_song", generate_song)
    with database.session() as session:
        session.add(
            Song(
                id=song_id,
                prompt="rain",
                duration_seconds=8,
                status=SongStatus.queued.value,
            )
        )
        session.commit()
        run_song(song_id, session, pipeline, settings)
        song = session.get(Song, song_id)

    assert song is not None
    assert song.status == SongStatus.completed.value
    assert song.lyrics == "Rain on the glass"
    assert song.tempo_bpm == 96
    assert song.duration_seconds == 10
    assert song.step_timings == {"generating_music": song.step_timings["generating_music"]}
    assert final_path(settings, song_id).read_bytes() == b"ID3-ace"
    assert not instrumental_path(settings, song_id).exists()


def test_generate_song_keeps_worker_timings(settings_factory, monkeypatch, tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    settings = settings_factory(
        music_generator="acestep",
        acestep_python=str(python),
        acestep_root=tmp_path / "ace",
        acestep_keep_warm=False,
    )
    output = tmp_path / "final.mp3"
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[bytes]:
        request = json.loads(Path(cmd[2]).read_text(encoding="utf-8"))
        seen["request"] = request
        Path(cmd[3]).write_text(
            json.dumps(
                {
                    "ok": True,
                    "lyrics": "line",
                    "bpm": 100,
                    "timings": {"process_start": 1.25, "total": 3.5},
                    "timing_details": {"loading_dit": 9.0},
                }
            ),
            encoding="utf-8",
        )
        output.write_bytes(b"ID3")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("songcraft.pipeline.ace_song.subprocess.run", fake_run)
    payload = AceStepGenerator(settings).generate_song("a gentle song", 30, output, job_id="job-9")

    request = seen["request"]
    assert isinstance(request, dict)
    assert request["job_id"] == "job-9"
    assert request["model"] == "acestep-v15-turbo"
    assert request["lm_model"] == "acestep-5Hz-lm-0.6B"
    timings = payload["timings"]
    assert isinstance(timings, dict)
    assert timings["process_start"] == 1.25
    assert timings["subprocess_wall"] >= 0
    assert payload["timing_details"] == {"loading_dit": 9.0}
    assert not output.with_suffix(".request.json").exists()
    assert not output.with_suffix(".result.json").exists()


def test_call_timer_keeps_the_wrapped_signature() -> None:
    class Handler:
        def generate_music(self, captions: str, lyrics: str = "") -> str:
            return captions + lyrics

    handler = Handler()
    timer = _CallTimer()
    restore = timer.install(handler, "generate_music")
    try:
        assert list(inspect.signature(handler.generate_music).parameters) == ["captions", "lyrics"]
        assert handler.generate_music("a", lyrics="b") == "ab"
        assert timer.seconds >= 0
    finally:
        restore()
    assert handler.generate_music("a") == "a"

    class Saver:
        def save_audio(self, path: str) -> str:
            return path

    save_timer = _CallTimer()
    restore_save = save_timer.install(Saver, "save_audio")
    try:
        assert Saver().save_audio("out.mp3") == "out.mp3"
        assert save_timer.seconds >= 0
    finally:
        restore_save()
