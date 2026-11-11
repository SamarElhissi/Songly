"""Time N songs through the real ACE-Step path (a fresh subprocess per song).

From the repository root, with the API virtualenv:

    .venv\\Scripts\\python scripts\\benchmark_generation.py
    .venv\\Scripts\\python scripts\\benchmark_generation.py --count 3 --duration 30

The summary at the end is the baseline to keep. Stage lines on stderr are the
same lines the API logs. This does not start the HTTP server. Songs are written
under data/benchmark/.
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# Settings reads .env from the current directory. The API is started from the repo root.
os.chdir(ROOT)

from songcraft.config import get_settings  # noqa: E402
from songcraft.pipeline.ace_song import AceStepGenerator, acestep_duration  # noqa: E402

# process_start through other sum to total. The last two are parent-clock rows.
WALL_STAGES = (
    "process_start",
    "loading_models",
    "planning_lyrics",
    "lm_audio_codes",
    "generating_audio",
    "saving_mp3",
    "other",
    "total",
    "subprocess_wall",
    "spawn_and_exit",
    "loading_models_total",
    "generating_music",
)

DEFAULT_PROMPT = "a gentle acoustic love song with soft vocals"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    if args.keep_warm is not None:
        settings = settings.model_copy(update={"acestep_keep_warm": args.keep_warm})
    generator = AceStepGenerator(settings)
    seconds = acestep_duration(args.duration)
    out_dir = Path(settings.data_dir) / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"ACE-Step benchmark  count={args.count}  duration={seconds:.0f}s  "
        f"keep_warm={settings.acestep_keep_warm}  device={settings.device}  prompt={args.prompt!r}"
    )
    if settings.acestep_keep_warm:
        print("The ACE-Step process stays up between songs.")
    else:
        print("Each song starts a new .venv-acestep process.")
    print()

    songs: list[dict[str, float]] = []
    details: list[dict[str, float]] = []
    try:
        for index in range(1, args.count + 1):
            job_id = f"bench-{index}"
            output = out_dir / f"{job_id}.mp3"
            print(f"--- {job_id} starting ---", flush=True)
            payload = generator.generate_song(args.prompt, seconds, output, job_id=job_id)
            timings = _with_spawn(_numbers(payload.get("timings")))
            if "generating_music" in payload and isinstance(payload["generating_music"], (int, float)):
                timings["generating_music"] = float(payload["generating_music"])
            if "loading_models" in payload and isinstance(payload["loading_models"], (int, float)):
                timings["loading_models_total"] = float(payload["loading_models"])
            detail = _numbers(payload.get("timing_details"))
            songs.append(timings)
            details.append(detail)
            _print_song(job_id, output, timings)
            print(flush=True)
    finally:
        generator.shutdown()

    print("=== wall clock (seconds) ===")
    print("Rows from process_start through other add up to total.")
    print("subprocess_wall is the parent clock. spawn_and_exit is subprocess_wall minus total.")
    _print_table(_stage_names(songs, WALL_STAGES), songs)
    print()
    print("=== detail (seconds, already inside the rows above) ===")
    print("loading_* is inside loading_models. ace_* is ACE-Step's own clock inside generation.")
    _print_table(_stage_names(details, ()), details)
    print()
    _print_warm_hint(songs)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ACE-Step song generation.")
    parser.add_argument("--count", type=int, default=3, help="Number of songs (default 3)")
    parser.add_argument("--duration", type=float, default=30, help="Song length in seconds (default 30)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to every song")
    parser.add_argument(
        "--keep-warm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override SONGCRAFT_ACESTEP_KEEP_WARM for this run",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def _numbers(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    numbers: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        numbers[str(key)] = float(item)
    return numbers


def _with_spawn(timings: dict[str, float]) -> dict[str, float]:
    """Process spawn and exit are in the parent clock but not in the worker total."""
    wall = timings.get("subprocess_wall")
    total = timings.get("total")
    if wall is None or total is None:
        return timings
    return {**timings, "spawn_and_exit": wall - total}


def _print_song(job_id: str, output: Path, timings: dict[str, float]) -> None:
    parts = [f"{job_id}  file={output}"]
    for stage in WALL_STAGES:
        if stage in timings:
            parts.append(f"{stage}={timings[stage]:.3f}")
    print(" ".join(parts))


def _stage_names(rows: list[dict[str, float]], preferred: tuple[str, ...]) -> list[str]:
    names = [stage for stage in preferred if any(stage in row for row in rows)]
    extra: list[str] = []
    for row in rows:
        for stage in row:
            if stage not in names and stage not in extra:
                extra.append(stage)
    return names + extra


def _print_table(stages: list[str], rows: list[dict[str, float]]) -> None:
    if not stages:
        print("(none)")
        return
    header = f"{'stage':<28} {'min':>8} {'median':>8} {'max':>8}"
    print(header)
    for stage in stages:
        values = [row[stage] for row in rows if stage in row]
        if not values:
            continue
        low = min(values)
        mid = statistics.median(values)
        high = max(values)
        print(f"{stage:<28} {low:8.3f} {mid:8.3f} {high:8.3f}")


def _print_warm_hint(songs: list[dict[str, float]]) -> None:
    """Point at the rows a warm process could skip. The fraction needs these numbers."""
    print("Cold every song. A warm process could skip process_start, loading_models,")
    print("and spawn_and_exit (subprocess_wall - total). It would still pay")
    print("planning_lyrics, lm_audio_codes, generating_audio, and saving_mp3.")


if __name__ == "__main__":
    raise SystemExit(main())
