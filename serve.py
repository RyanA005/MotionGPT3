"""
MotionGPT3 HTTP API - also the server entrypoint:

  python serve.py

  uvicorn serve:app --host 127.0.0.1 --port 8888
"""

from __future__ import annotations

import json
import random
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.device import resolve_torch_device

_model = None
_cfg = None


def load_model_once() -> None:
    """Load checkpoint once; safe to call multiple times."""
    global _model, _cfg
    if _model is not None:
        return
    _argv = sys.argv
    sys.argv = [_argv[0]] if _argv else ["python"]
    try:
        _cfg = parse_args(phase="webui")
    finally:
        sys.argv = _argv
    _cfg.FOLDER = "cache"
    Path(_cfg.FOLDER).mkdir(parents=True, exist_ok=True)
    pl.seed_everything(_cfg.SEED_VALUE)
    device = resolve_torch_device(_cfg.ACCELERATOR)
    datamodule = build_data(_cfg, phase="test")
    _model = build_model(_cfg, datamodule).eval()
    state_dict = torch.load(_cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    _model.load_state_dict(state_dict)
    _model.to(device)


def get_model():
    return _model


def set_inference_seed(seed: int) -> None:
    """Set deterministic seeds for one inference request."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def generate_motion(
    prompt: str,
    motion_length: int = 0,
    motion_seconds: float = 0.0,
    seed: int | None = None,
    task: Literal["t2m", "t2t", "m2t"] = "t2m",
    include_feats: bool = False,
    motion_npy: str | None = None,
) -> dict[str, Any]:
    """Run a single forward pass. For ``m2t``, pass ``motion_npy`` (HumanML 263-d features)."""
    if _model is None:
        raise RuntimeError("Model not loaded; call load_model_once() first.")
    if seed is not None:
        set_inference_seed(int(seed))
    model = get_model()
    fps = float(model.fps) if model is not None else 20.0
    length_frames = int(round(motion_seconds * fps)) if motion_seconds > 0 else int(motion_length)
    lm = _model.lm
    fulfilled = lm.placeholder_fulfill(
        prompt, length_frames, lm.input_motion_holder_seq, ""
    )
    batch: dict[str, Any] = {
        "length": [length_frames],
        "text": [fulfilled],
        "motion_tokens_input": None,
        "feats_ref": None,
    }

    if task == "m2t":
        if not motion_npy:
            raise ValueError(
                "task=m2t requires motion_npy= path to a .npy array of shape (T, 263) "
                "(HumanML3D feature vectors)."
            )
        feats_np = np.load(motion_npy)
        feats_t = torch.from_numpy(feats_np.astype(np.float32)).to(_model.device)
        mlen = int(feats_t.shape[0])
        feats_t = _model.datamodule.normalize(feats_t.unsqueeze(0))
        batch["motion_tokens_input"] = _model.lm.motion_feats_to_tokens(
            _model.vae, feats_t, [mlen], modes="m2t"
        )
        batch["length"] = [mlen]

    out = _model(batch, task=task)
    if task in ("t2t", "m2t"):
        return {
            "task": task,
            "text": out["texts"][0],
            "motion_length_frames": 0,
            "joints": None,
            "feats": None,
        }
    out_lengths = int(out["length"][0])
    out_joints = out["joints"][:out_lengths].detach().cpu().numpy()
    out_texts = out["texts"][0]
    result: dict[str, Any] = {
        "task": task,
        "text": out_texts,
        "motion_length_frames": out_lengths,
        "joints": out_joints.tolist(),
    }
    if include_feats:
        result["feats"] = out["feats"][0].detach().cpu().numpy().tolist()
    return result


def generate_artifacts(
    prompt: str,
    motion_length: int = 0,
    motion_seconds: float = 0.0,
    seed: int | None = None,
    task: Literal["t2m", "t2t", "m2t"] = "t2m",
    motion_npy: str | None = None,
    output_dir: str = "cache",
    skeleton: bool = False,
    full_response: bool = False,
) -> dict[str, Any]:
    """Generate motion, write JSON (+ MP4 for t2m), return summary or full payload."""
    load_model_once()
    out = generate_motion(
        prompt,
        motion_length=motion_length,
        motion_seconds=motion_seconds,
        seed=seed,
        task=task,
        include_feats=True,
        motion_npy=motion_npy,
    )
    stem = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + str(
        np.random.randint(10000, 99999)
    )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if task == "t2m" and out.get("joints") is not None:
        from motGPT.utils.render_utils import render_motion

        joints = np.asarray(out["joints"], dtype=np.float32)
        m = get_model()
        fps = float(m.fps) if m is not None else 20.0
        method = "fast" if skeleton else "slow"
        render_motion(joints, joints, str(out_path), fname=stem, method=method, fps=fps)
        out["video_path"] = str((out_path / f"{stem}.mp4").resolve())

    json_file = out_path / f"{stem}.json"
    out["json_path"] = str(json_file.resolve())
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    if full_response:
        return out
    summary: dict[str, Any] = {
        "task": out["task"],
        "text": out["text"],
        "motion_length_frames": out.get("motion_length_frames"),
        "json_path": out["json_path"],
    }
    if "video_path" in out:
        summary["video_path"] = out["video_path"]
    return summary


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    motion_length: int = Field(0, ge=0)
    motion_seconds: float = Field(0.0, ge=0.0)
    seed: int | None = Field(None, ge=0)
    task: Literal["t2m", "t2t", "m2t"] = "t2m"
    include_feats: bool = False
    motion_npy: str | None = Field(
        None,
        description="Path to (T,263) features .npy (required for m2t).",
    )


class GenerateArtifactsBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    motion_length: int = Field(0, ge=0)
    motion_seconds: float = Field(0.0, ge=0.0)
    seed: int | None = Field(None, ge=0)
    task: Literal["t2m", "t2t", "m2t"] = "t2m"
    motion_npy: str | None = Field(
        None,
        description="Path to (T,263) features .npy (required for m2t).",
    )
    output_dir: str = "cache"
    skeleton: bool = False
    full_response: bool = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model_once()
    yield


app = FastAPI(title="MotionGPT3", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/generate")
def post_generate(body: GenerateBody) -> dict[str, Any]:
    return generate_motion(
        body.prompt,
        motion_length=body.motion_length,
        motion_seconds=body.motion_seconds,
        seed=body.seed,
        task=body.task,
        include_feats=body.include_feats,
        motion_npy=body.motion_npy,
    )


@app.post("/generate/artifacts")
def post_generate_artifacts(body: GenerateArtifactsBody) -> dict[str, Any]:
    """Write timestamped JSON (+ MP4 for t2m) under output_dir; response is a short summary by default."""
    return generate_artifacts(
        body.prompt,
        motion_length=body.motion_length,
        motion_seconds=body.motion_seconds,
        seed=body.seed,
        task=body.task,
        motion_npy=body.motion_npy,
        output_dir=body.output_dir,
        skeleton=body.skeleton,
        full_response=body.full_response,
    )


if __name__ == "__main__":
    import os

    if "127.0.0.1" not in os.environ.get("NO_PROXY", ""):
        os.environ["NO_PROXY"] = (
            os.environ.get("NO_PROXY", "") + ",127.0.0.1,localhost"
        ).strip(",").strip()

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8888)
