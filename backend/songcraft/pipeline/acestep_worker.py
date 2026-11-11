"""Run inside .venv-acestep. Reads a request json path and writes a result json path.

Stage lines go to stderr as `stage_finished job_id=... stage=... duration_s=...`.
They are measurement only. Model choice, prompts, and generation settings stay the same.
"""

import functools
import gc
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        return serve()
    origin = time.perf_counter()
    request_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    job_id = str(request.get("job_id") or "-")
    timings: dict[str, float] = {}
    details: dict[str, float] = {}
    try:
        payload = _generate(request, job_id, origin, timings, details)
    except Exception as exc:
        traceback.print_exc()
        payload = {"ok": False, "error": str(exc)}
    _finish_timings(job_id, origin, timings)
    payload["timings"] = timings
    payload["timing_details"] = details
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload.get("ok") else 1


def _generate(
    request: dict,
    job_id: str,
    origin: float,
    timings: dict[str, float],
    details: dict[str, float],
) -> dict:
    runtime = _load_models(request, job_id, origin, timings, details)
    return _render(runtime, request, job_id, timings, details)


# Replies use this stream. serve() points sys.stdout at stderr so ACE-Step's
# own prints, such as "Using precomputed LM hints", cannot break the protocol.
_protocol = sys.stdout


def serve() -> int:
    """Handle one JSON command per stdin line. Models stay loaded until shutdown."""
    global _protocol
    _protocol = sys.stdout
    sys.stdout = sys.stderr
    runtime: _Runtime | None = None
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
        except json.JSONDecodeError:
            _reply({"op": "result", "ok": False, "error": "ACE-Step worker stopped unexpectedly"})
            continue
        if not isinstance(request, dict):
            _reply({"op": "result", "ok": False, "error": "ACE-Step worker stopped unexpectedly"})
            continue
        op = request.get("op")
        job_id = str(request.get("job_id") or "-")
        if op == "shutdown":
            _reply({"op": "bye", "ok": True})
            return 0
        if op == "load":
            timings: dict[str, float] = {}
            details: dict[str, float] = {}
            try:
                runtime = _load_models(request, job_id, time.perf_counter(), timings, details)
            except Exception as exc:
                traceback.print_exc()
                _reply({"op": "ready", "job_id": job_id, "ok": False, "error": str(exc)})
                return 1
            _reply(
                {
                    "op": "ready",
                    "job_id": job_id,
                    "ok": True,
                    "loading_models": timings.get("loading_models", 0.0),
                }
            )
            continue
        if op == "generate":
            if runtime is None:
                _reply(
                    {
                        "op": "result",
                        "job_id": job_id,
                        "ok": False,
                        "error": "ACE-Step worker stopped unexpectedly",
                    }
                )
                continue
            timings = {}
            details = {}
            try:
                payload = _render(runtime, request, job_id, timings, details)
            except Exception as exc:
                traceback.print_exc()
                _release_cuda()
                fatal = _is_oom(exc)
                _reply(
                    {
                        "op": "result",
                        "job_id": job_id,
                        "ok": False,
                        "error": "ACE-Step ran out of GPU memory" if fatal else str(exc),
                        "fatal": fatal,
                    }
                )
                if fatal:
                    return 1
                continue
            _release_cuda()
            payload["op"] = "result"
            payload["job_id"] = job_id
            _reply(payload)
            continue
        _reply({"op": "result", "job_id": job_id, "ok": False, "error": "ACE-Step worker stopped unexpectedly"})
    return 0


@dataclass
class _Runtime:
    dit: object
    llm: object
    audio_saver: type
    generation_params: type
    generation_config: type
    create_sample: Callable
    generate_music: Callable


def _load_models(
    request: dict,
    job_id: str,
    origin: float,
    timings: dict[str, float],
    details: dict[str, float],
) -> _Runtime:
    try:
        from acestep.audio_utils import AudioSaver
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
        from acestep.llm_inference import LLMHandler
        from acestep.model_downloader import ensure_lm_model, ensure_main_model
    except Exception:
        _add_stage(job_id, timings, "process_start", origin)
        raise
    # Python startup plus the ACE-Step import. Weight loading is the next stage.
    _add_stage(job_id, timings, "process_start", origin)

    project_root = request["project_root"]
    checkpoint_dir = request["checkpoint_dir"]
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    checkpoint_started = load_started
    ready, message = ensure_main_model(checkpoint_dir)
    if not ready:
        raise RuntimeError(message)
    ready, message = ensure_lm_model(request["lm_model"], checkpoint_dir)
    if not ready:
        raise RuntimeError(message)
    _add_stage(job_id, details, "checking_checkpoints", checkpoint_started)

    dit = AceStepHandler()
    dit_started = time.perf_counter()
    status, ok = dit.initialize_service(
        project_root=project_root,
        config_path=request["model"],
        device=request["device"],
        use_flash_attention=False,
        compile_model=True,
        offload_to_cpu=True,
        offload_dit_to_cpu=True,
        quantization="int8_weight_only",
        use_mlx_dit=False,
    )
    if not ok:
        raise RuntimeError(status)
    # Includes reading the turbo weights, int8 load, and torch.compile.
    _add_stage(job_id, details, "loading_dit", dit_started)

    llm = LLMHandler()
    lm_started = time.perf_counter()
    status, ok = llm.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=request["lm_model"],
        backend="pt",
        device=request["device"],
        offload_to_cpu=True,
    )
    if not ok:
        raise RuntimeError(status)
    _add_stage(job_id, details, "loading_lm", lm_started)
    _add_stage(job_id, timings, "loading_models", load_started)
    return _Runtime(dit, llm, AudioSaver, GenerationParams, GenerationConfig, create_sample, generate_music)


def _render(
    runtime: _Runtime,
    request: dict,
    job_id: str,
    timings: dict[str, float],
    details: dict[str, float],
) -> dict:
    plan_started = time.perf_counter()
    sample = runtime.create_sample(
        runtime.llm,
        query=request["prompt"],
        instrumental=False,
        vocal_language="en",
    )
    _add_stage(job_id, timings, "planning_lyrics", plan_started)
    if not sample.success:
        raise RuntimeError(sample.error or sample.status_message or "ACE-Step could not plan the song")

    output_path = Path(request["output_path"])
    save_dir = output_path.parent / f".{output_path.stem}-ace"
    save_dir.mkdir(parents=True, exist_ok=True)
    params = runtime.generation_params(
        task_type="text2music",
        caption=sample.caption or request["prompt"],
        lyrics=sample.lyrics,
        instrumental=False,
        vocal_language=sample.language or "en",
        bpm=sample.bpm,
        keyscale=sample.keyscale or "",
        timesignature=sample.timesignature or "",
        duration=float(request["duration_seconds"]),
        inference_steps=8,
        guidance_scale=1.0,
        shift=3.0,
        infer_method="ode",
        thinking=True,
    )
    config = runtime.generation_config(batch_size=1, audio_format="mp3")
    # thinking=True calls the 0.6B model again inside generate_music, for audio
    # codes, before the turbo decoder. Time that call on its own.
    # save_audio also runs inside generate_music; time it on its own too.
    # Wrappers only record duration and then call the original.
    lm_timer = _CallTimer()
    dit_timer = _CallTimer()
    save_timer = _CallTimer()
    restore = [
        lm_timer.install(runtime.llm, "generate_with_stop_condition"),
        # functools.wraps keeps the real signature. generate_music inspects it
        # and would drop every argument if the wrapper hid the parameters.
        dit_timer.install(runtime.dit, "generate_music"),
        save_timer.install(runtime.audio_saver, "save_audio"),
    ]
    generation_error: BaseException | None = None
    try:
        result = runtime.generate_music(runtime.dit, runtime.llm, params, config, save_dir=str(save_dir))
    except BaseException as exc:
        generation_error = exc
        result = None
    finally:
        for undo in reversed(restore):
            undo()
    if generation_error is not None or result is None or not result.success:
        _record_generation(job_id, timings, lm_timer.seconds, dit_timer.seconds, save_timer.seconds)
        if generation_error is not None:
            raise generation_error
        detail = getattr(result, "error", None) or getattr(result, "status_message", None)
        raise RuntimeError(str(detail or "ACE-Step generation failed"))
    _copy_time_costs(job_id, details, getattr(result, "extra_outputs", None))

    audio = result.audios[0]
    produced = Path(str(audio["path"]))
    copy_started = time.perf_counter()
    if produced.resolve() != output_path.resolve():
        shutil.copyfile(produced, output_path)
    copy_elapsed = time.perf_counter() - copy_started
    _record_generation(
        job_id,
        timings,
        lm_timer.seconds,
        dit_timer.seconds,
        save_timer.seconds + copy_elapsed,
    )
    shutil.rmtree(save_dir, ignore_errors=True)
    bpm = getattr(result, "bpm", None)
    if bpm is None:
        bpm = audio.get("bpm")
    return {
        "ok": True,
        "lyrics": sample.lyrics,
        "bpm": sample.bpm if sample.bpm is not None else bpm,
    }


class _CallTimer:
    """Accumulates wall time spent inside one wrapped callable."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def install(self, obj: object, name: str) -> Callable[[], None]:
        original = getattr(obj, name)
        timer = self

        @functools.wraps(original)
        def wrapped(*args: object, **kwargs: object) -> object:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timer.seconds += time.perf_counter() - started

        setattr(obj, name, wrapped)

        def restore() -> None:
            setattr(obj, name, original)

        return restore


def _copy_time_costs(job_id: str, details: dict[str, float], extra: object) -> None:
    """ACE-Step's own clocks, nested inside generating_audio. Not added to the total."""
    if not isinstance(extra, dict):
        return
    costs = extra.get("time_costs")
    if not isinstance(costs, dict):
        return
    for key, value in costs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        _add_seconds(job_id, details, f"ace_{key}", float(value))


def _record_generation(
    job_id: str,
    timings: dict[str, float],
    lm_seconds: float,
    dit_seconds: float,
    save_seconds: float,
) -> None:
    _add_seconds(job_id, timings, "lm_audio_codes", lm_seconds)
    _add_seconds(job_id, timings, "generating_audio", dit_seconds)
    _add_seconds(job_id, timings, "saving_mp3", save_seconds)


def _reply(payload: dict) -> None:
    _protocol.write(json.dumps(payload) + "\n")
    _protocol.flush()


def _release_cuda() -> None:
    """Drop cached GPU blocks between songs. Weights stay loaded."""
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def _add_stage(job_id: str, bucket: dict[str, float], stage: str, started: float) -> None:
    _add_seconds(job_id, bucket, stage, time.perf_counter() - started)


def _add_seconds(job_id: str, bucket: dict[str, float], stage: str, seconds: float) -> None:
    bucket[stage] = round(seconds, 3)
    print(
        f"stage_finished job_id={job_id} stage={stage} duration_s={seconds:.3f}",
        file=sys.stderr,
        flush=True,
    )


def _finish_timings(job_id: str, origin: float, timings: dict[str, float]) -> None:
    """Fill the gap so the named stages plus `other` equal `total`."""
    elapsed = time.perf_counter() - origin
    accounted = sum(timings.values())
    other = max(0.0, elapsed - accounted)
    _add_seconds(job_id, timings, "other", other)
    _add_seconds(job_id, timings, "total", elapsed)


if __name__ == "__main__":
    sys.exit(main())
