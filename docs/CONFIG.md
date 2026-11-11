# Configuration

Copy `.env.example` to `.env` at the repository root. Settings use the `SONGCRAFT_` prefix (`backend/songcraft/config.py`).

## Default path (ACE-Step or fake)

| Variable | Default | Role |
| --- | --- | --- |
| `SONGCRAFT_DATABASE_URL` | `sqlite:///./data/songcraft.db` | SQLite file |
| `SONGCRAFT_DATA_DIR` | `./data` | Parent of the DB and `songs/` audio |
| `SONGCRAFT_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. For ACE-Step, `auto` is sent to the worker as `cuda`. |
| `SONGCRAFT_MIN_DURATION_SECONDS` | `5` | Lower clamp for `POST /api/songs` |
| `SONGCRAFT_MAX_DURATION_SECONDS` | `30` | Upper clamp for `POST /api/songs` |
| `SONGCRAFT_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins |
| `SONGCRAFT_MUSIC_GENERATOR` | `acestep` | `acestep` or `fake` |
| `SONGCRAFT_ACESTEP_PYTHON` | _(empty)_ | Python executable for the ACE-Step worker. Empty means `<repo>/.venv-acestep/Scripts/python.exe` |
| `SONGCRAFT_ACESTEP_ROOT` | `./ace-step` | ACE-Step checkout (checkpoints live under `<root>/checkpoints`) |
| `SONGCRAFT_ACESTEP_KEEP_WARM` | `true` | Reuse one ACE-Step process across songs. `false` starts a fresh process per song |
| `SONGCRAFT_ACESTEP_IDLE_SECONDS` | `600` | Exit the warm process after this many idle seconds. `0` keeps it until the API stops |
| `SONGCRAFT_ACESTEP_SONG_TIMEOUT_SECONDS` | `300` | Stop a song that is still running after this many seconds. The next song starts a new process |
| `SONGCRAFT_JOB_QUEUE` | `inprocess` | `inprocess` (background thread) or `inline` (tests) |

Set `SONGCRAFT_MUSIC_GENERATOR=fake` (and the older-pipeline slots to `fake` if you want zero model use) to run without ACE-Step or Hugging Face downloads.

ACE-Step accepts durations from 10 to 600 seconds inside the worker. Values below 10 from the API are raised to 10 before the call (`acestep_duration` in `ace_song.py`). The form slider is 10–30 seconds.

## Optional older pipeline

Used when `SONGCRAFT_MUSIC_GENERATOR` is not `acestep` (today that means `fake` music plus real or fake lyrics/vocals/mixer). Ignored while `MUSIC_GENERATOR=acestep`.

| Variable | Default | Role |
| --- | --- | --- |
| `SONGCRAFT_LYRICS_GENERATOR` | `instruct` | `instruct`, `gpt2`, or `fake` |
| `SONGCRAFT_VOCAL_SYNTHESIZER` | `melody` | `melody`, `gtts`, or `fake` |
| `SONGCRAFT_MIXER` | `pydub` | `pydub` or `fake` |
| `SONGCRAFT_BEAT_ANALYZER` | `librosa` | `librosa` or `fake` |
| `SONGCRAFT_LYRICS_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face id for `instruct` or `gpt2` |

The pydub mixer needs `ffmpeg` and `ffprobe` on `PATH`, or `ffmpeg/ffmpeg.exe` and `ffmpeg/ffprobe.exe` in the repo.
