import json
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from songcraft.config import Settings
from songcraft.pipeline.warm_worker import AceStepWorkerError, WarmWorker

logger = logging.getLogger("songcraft.pipeline")

# ACE-Step accepts 10 to 600 seconds. Shorter slider values are lifted to the minimum.
_MIN_SECONDS = 10.0
_MAX_SECONDS = 600.0


def acestep_duration(seconds: float) -> float:
    return min(_MAX_SECONDS, max(_MIN_SECONDS, float(seconds)))


class AceStepGenerator:
    """Generate a finished song, vocals included, with ACE-Step 1.5.

    The model needs its own PyTorch build, so generation runs in .venv-acestep
    instead of the API virtualenv.
    """

    def __init__(self, settings: Settings, *, command: list[str] | None = None) -> None:
        self._settings = settings
        self._command = command
        self._warm: WarmWorker | None = None

    def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
        raise RuntimeError("ACE-Step writes the full mix, not a separate instrumental")

    def generate_song(
        self,
        prompt: str,
        duration_seconds: float,
        output_path: Path,
        *,
        job_id: str | None = None,
        on_loaded: Callable[[float], None] | None = None,
    ) -> dict[str, object]:
        seconds = acestep_duration(duration_seconds)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = _song_request(self._settings, prompt, seconds, output_path, job_id)
        if self._settings.acestep_keep_warm:
            return self._generate_warm(request, output_path, seconds, on_loaded)
        return self._generate_cold(request, output_path, seconds)

    def shutdown(self) -> None:
        if self._warm is not None:
            self._warm.shutdown()

    def worker_pid(self) -> int | None:
        if self._warm is None:
            return None
        return self._warm.worker_pid()

    def _generate_warm(
        self,
        request: dict[str, object],
        output_path: Path,
        seconds: float,
        on_loaded: Callable[[float], None] | None,
    ) -> dict[str, object]:
        if self._warm is None:
            self._warm = WarmWorker(
                self._serve_command(),
                idle_seconds=self._settings.acestep_idle_seconds,
                song_timeout_seconds=self._settings.acestep_song_timeout_seconds,
            )
        payload, loading, generating = self._warm.run_song(request, on_loaded=on_loaded)
        if not output_path.is_file():
            raise AceStepWorkerError("ACE-Step reported success but the audio file is missing")
        payload["duration"] = seconds
        payload["generating_music"] = round(generating, 3)
        if loading is not None:
            payload["loading_models"] = round(loading, 3)
        logger.info(
            "stage_finished job_id=%s stage=generating_music duration_s=%.3f",
            request.get("job_id") or "-",
            generating,
        )
        return payload

    def _generate_cold(self, request: dict[str, object], output_path: Path, seconds: float) -> dict[str, object]:
        python = _ace_python(self._settings)
        if not python.is_file():
            raise AceStepWorkerError(
                f"ACE-Step Python was not found at {python}. "
                "Create .venv-acestep and install ACE-Step 1.5 into it."
            )
        request_path = output_path.with_suffix(".request.json")
        result_path = output_path.with_suffix(".result.json")
        request_path.write_text(json.dumps(request), encoding="utf-8")
        worker = Path(__file__).with_name("acestep_worker.py")
        job_id = request.get("job_id") or "-"
        logger.info("stage_started job_id=%s stage=subprocess", job_id)
        started = time.perf_counter()
        completed = subprocess.run(
            [str(python), str(worker), str(request_path), str(result_path)],
            check=False,
        )
        wall = time.perf_counter() - started
        logger.info(
            "stage_finished job_id=%s stage=subprocess_wall duration_s=%.3f",
            job_id,
            wall,
        )
        if not result_path.is_file():
            raise AceStepWorkerError(f"ACE-Step did not write a result (exit {completed.returncode})")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        request_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        timings = payload.get("timings")
        if isinstance(timings, dict):
            timings["subprocess_wall"] = round(wall, 3)
        if not payload.get("ok"):
            raise AceStepWorkerError(_clean_error(payload.get("error")))
        if not output_path.is_file():
            raise AceStepWorkerError("ACE-Step reported success but the audio file is missing")
        loading, generating = _cold_step_times(timings, wall)
        if loading is not None:
            payload["loading_models"] = loading
        payload["generating_music"] = generating
        payload["duration"] = seconds
        return payload

    def _serve_command(self) -> list[str]:
        if self._command is not None:
            return self._command
        python = _ace_python(self._settings)
        if not python.is_file():
            raise AceStepWorkerError(
                f"ACE-Step Python was not found at {python}. "
                "Create .venv-acestep and install ACE-Step 1.5 into it."
            )
        worker = Path(__file__).with_name("acestep_worker.py")
        return [str(python), "-u", str(worker), "--serve"]


def _ace_root(settings: Settings) -> Path:
    configured = Path(settings.acestep_root)
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[3] / configured


def _ace_python(settings: Settings) -> Path:
    if settings.acestep_python:
        return Path(settings.acestep_python)
    return Path(__file__).resolve().parents[3] / ".venv-acestep" / "Scripts" / "python.exe"


def _song_request(
    settings: Settings,
    prompt: str,
    seconds: float,
    output_path: Path,
    job_id: str | None,
) -> dict[str, object]:
    root = _ace_root(settings)
    request: dict[str, object] = {
        "prompt": prompt,
        "duration_seconds": seconds,
        "output_path": str(output_path),
        "project_root": str(root),
        "checkpoint_dir": str(root / "checkpoints"),
        "device": "cuda" if settings.device == "auto" else settings.device,
        "model": "acestep-v15-turbo",
        "lm_model": "acestep-5Hz-lm-0.6B",
    }
    if job_id:
        request["job_id"] = job_id
    return request


def _cold_step_times(timings: object, wall: float) -> tuple[float | None, float]:
    """Map a one-shot run onto loading_models plus generating_music."""
    if not isinstance(timings, dict):
        return None, round(wall, 3)
    loading = timings.get("loading_models")
    startup = timings.get("process_start")
    total = timings.get("total")
    if not isinstance(loading, (int, float)):
        return None, round(wall, 3)
    ready = float(loading)
    if isinstance(startup, (int, float)):
        ready += float(startup)
    if isinstance(total, (int, float)):
        ready += max(0.0, wall - float(total))
    return round(ready, 3), round(max(0.0, wall - ready), 3)


def _clean_error(error: object) -> str:
    text = " ".join(str(error or "ACE-Step failed").split())
    if len(text) > 300:
        return text[:300].rstrip() + "…"
    return text
