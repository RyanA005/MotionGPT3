# MotionGPT3 (inference)

Text-to-motion and related tasks using a single FastAPI server. Training code (`train.py`) remains for full model training; day-to-day use is `**serve.py**` plus HTTP.

**Setup, venv, downloads, and GPU notes:** see [INSTALLATION.md](INSTALLATION.md).

## What was trimmed

- **Gradio / old web UI** — removed; use the HTTP API instead.
- **Extra launchers** — one entrypoint: `python serve.py` (no separate `app.py` / `api.py` shims).
- **Standalone `fit.py`** (joints→SMPL offline tool) — removed as unused by inference.
- **Slimmer inference path** — metrics/evaluators can be skipped at load via config where applicable; dependencies are oriented around inference and rendering.

## Run the server

```bash
python serve.py
```

Listens on **[http://127.0.0.1:8888](http://127.0.0.1:8888)** (see `serve.py` for changing host/port in the `__main__` block).

Optional helper (same machine, server already up):

```bash
bash prompt.sh "a person walks"
```

## API


| Method | Path                  | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------ | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health`             | Liveness: `status`, whether the model finished loading.                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `POST` | `/generate`           | One forward pass. JSON body: `prompt`, optional `motion_length`, `task` (`t2m` | `t2t` | `m2t`), `include_feats`, `motion_npy` (required for `m2t`, path on the **server** filesystem). Returns model output (e.g. joints as lists for `t2m`).                                                                                                                                                                                                                                                      |
| `POST` | `/generate/artifacts` | Same generation with `**include_feats` forced on**, then writes a timestamped `**.json`** under `output_dir` (default `cache/`), and for `**t2m**` a `**.mp4**` (SMPL mesh by default, or set `skeleton: true` for a fast stick-figure). Response defaults to a **short summary** (`json_path`, `video_path`, etc.); set `full_response: true` to return the full dict in the HTTP body. Body: `prompt`, optional `motion_length`, `task`, `motion_npy`, `output_dir`, `skeleton`, `full_response`. |


All `POST` bodies are JSON (`Content-Type: application/json`).