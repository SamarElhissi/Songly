"""Speaks the warm-worker protocol without loading ACE-Step. Tests launch this process."""

import json
import sys
import time
from pathlib import Path


def main() -> int:
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        request = json.loads(text)
        op = request.get("op")
        job_id = request.get("job_id")
        if op == "shutdown":
            reply({"op": "bye", "ok": True})
            return 0
        if op == "load":
            reply({"op": "ready", "job_id": job_id, "ok": True, "loading_models": 0.01})
            continue
        if op == "generate":
            prompt = str(request.get("prompt") or "")
            if prompt == "crash":
                raise SystemExit(1)
            if prompt == "hang":
                time.sleep(30)
                continue
            if prompt == "noisy":
                print("Using precomputed LM hints", flush=True)
            if prompt == "oom":
                reply(
                    {
                        "op": "result",
                        "job_id": job_id,
                        "ok": False,
                        "error": "ACE-Step ran out of GPU memory",
                        "fatal": True,
                    }
                )
                return 1
            Path(str(request["output_path"])).write_bytes(b"ID3-fake")
            reply(
                {
                    "op": "result",
                    "job_id": job_id,
                    "ok": True,
                    "lyrics": "fake line",
                    "bpm": 90,
                }
            )
            continue
        reply({"op": "result", "job_id": job_id, "ok": False, "error": "unknown op"})
    return 0


def reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
