# Installation

**Python 3.10+**, **Git**, **Git LFS**, and **bash** 

```bash
pyt -3.10 -m venv .venv
source .venv/bin/activate
bash prepare/setup_all.sh
```

## Inference

1. **Start the API** (loads the model once):

   ```bash
   python serve.py
   ```

2. **Generate with artifacts** (writes under `cache/`): in another terminal, with the same venv activated:

   ```bash
   bash prompt.sh "a person walks"
   ```

   `prompt.sh` only `curl`s `http://127.0.0.1:8888` — the server must already be running. Override the base URL with `SERVER` if needed, e.g. `SERVER=http://127.0.0.1:9000 bash prompt.sh "hello"`.
