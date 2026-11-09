import json
import subprocess
from pathlib import Path

from songcraft.config import Settings

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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def generate(self, prompt: str, duration_seconds: float, output_path: Path) -> None:
        raise RuntimeError("ACE-Step writes the full mix, not a separate instrumental")

    def generate_song(self, prompt: str, duration_seconds: float, output_path: Path) -> dict[str, object]:
        seconds = acestep_duration(duration_seconds)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        root = _ace_root(self._settings)
        python = _ace_python(self._settings)
        if not python.is_file():
            raise RuntimeError(
                f"ACE-Step Python was not found at {python}. "
                "Create .venv-acestep and install ACE-Step 1.5 into it."
            )
        request_path = output_path.with_suffix(".request.json")
        result_path = output_path.with_suffix(".result.json")
        request_path.write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "duration_seconds": seconds,
                    "output_path": str(output_path),
                    "project_root": str(root),
                    "checkpoint_dir": str(root / "checkpoints"),
                    "device": "cuda" if self._settings.device == "auto" else self._settings.device,
                    "model": "acestep-v15-turbo",
                    "lm_model": "acestep-5Hz-lm-0.6B",
                }
            ),
            encoding="utf-8",
        )
        worker = Path(__file__).with_name("acestep_worker.py")
        completed = subprocess.run(
            [str(python), str(worker), str(request_path), str(result_path)],
            check=False,
        )
        if not result_path.is_file():
            raise RuntimeError(f"ACE-Step did not write a result (exit {completed.returncode})")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        request_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") or "ACE-Step failed"))
        if not output_path.is_file():
            raise RuntimeError("ACE-Step reported success but the audio file is missing")
        payload["duration"] = seconds
        return payload


def _ace_root(settings: Settings) -> Path:
    configured = Path(settings.acestep_root)
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[3] / configured


def _ace_python(settings: Settings) -> Path:
    if settings.acestep_python:
        return Path(settings.acestep_python)
    return Path(__file__).resolve().parents[3] / ".venv-acestep" / "Scripts" / "python.exe"
