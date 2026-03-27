import time
import os
from os.path import join as pjoin
import numpy as np
from pathlib import Path

import torch
import pytorch_lightning as pl
from pydantic_core.core_schema import ValidationInfo

from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.config import get_module_config
from omegaconf import OmegaConf

import moviepy.editor as mp
from scipy.spatial.transform import Rotation as RRR
from motGPT.render.pyrender.hybrik_loc2rot import HybrIKJointsToRotmat
from motGPT.render.pyrender.smpl_render import SMPLRender

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, field_validator

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    feats: np.ndarray
    lengths: int
    joints: np.ndarray
    texts: str

    model_config = ConfigDict(arbitrary_types_allowed=True, json_encoders={np.ndarray: lambda x: x.tolist()})

    @field_validator("feats", "joints", mode="before")
    @classmethod
    def ensure_numpy_array(cls, value: any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        try:
            return np.array(value, dtype=float)
        except Exception as e:
            raise ValueError(f"Cannot convert {value} to np.ndarray: {e}")

    @field_validator("lengths", mode="before")
    @classmethod
    def check_length(cls, value: int, info: ValidationInfo) -> int:
        """
        Ensure that 'lengths' matches feats.shape[0].
        In Pydantic v2, info.data contains other fields already processed.
        """
        feats = info.data.get("feats")  # <-- use info.data instead of values
        if feats is not None:
            if value != feats.shape[0]:
                raise ValueError(f"'lengths' ({value}) does not match feats shape {feats.shape}")
        return value

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        for key in ["feats", "joints"]:
            if isinstance(data.get(key), np.ndarray):
                data[key] = data[key].tolist()
        return data

    def __repr__(self):
        return (
            f"GenerateResponse(feats.shape={self.feats.shape}, "
            f"lengths={self.lengths}, "
            f"joints.shape={self.joints.shape}, "
            f"texts='{self.texts}')"
        )

def load_model():
    global model, cfg

    if cfg is None:
        OmegaConf.register_new_resolver("eval", eval)
        cfg_assets = OmegaConf.load("./configs/assets.yaml")
        cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
        cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("./configs/webui.yaml"))
        if not cfg_exp.FULL_CONFIG:
            cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
        cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    if model is None:
        cfg.FOLDER = 'cache'
        output_dir = Path(cfg.FOLDER)
        output_dir.mkdir(parents=True, exist_ok=True)
        pl.seed_everything(cfg.SEED_VALUE)
        if cfg.ACCELERATOR == "gpu":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        datamodule = build_data(cfg, phase="test")

        model = build_model(cfg, datamodule).eval()
        state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device)

def close_model():
    global model, cfg
    del model, cfg
    cfg, model = None, None
    torch.cuda.empty_cache()

cfg = None
model = None
motion_length = 0

app = FastAPI()

@app.post("/generate/")
def generate_motion(request: GenerateRequest) -> GenerateResponse:
    load_model()

    # Fulfill the prompt with placeholders for motion generation
    prompt = model.lm.placeholder_fulfill(request.prompt, motion_length, model.lm.input_motion_holder_seq, "")

    # Create a batch dictionary to feed into the model
    batch = {
        "length": [motion_length],
        "text": [prompt],
        "motion_tokens_input": None,
        "feats_ref": None,
    }

    # Generate motion using the model
    output = model(batch, task="t2m")

    # Extract the generated motion features, lengths, joints, and texts from the output
    out_feats = output["feats"][0].to('cpu').numpy()
    out_lengths = output["length"][0]
    out_joints = output["joints"][:out_lengths].detach().cpu().numpy()
    out_texts = output["texts"][0]

    close_model()
    return GenerateResponse(feats=out_feats, lengths=out_lengths, joints=out_joints, texts=out_texts)

def save_motion_as_video(joints, feats, save_dir):
    """
    Save the generated motion as a video file.

    Args:
        joints (numpy.ndarray): The generated joint positions.
        feats (numpy.ndarray): The generated motion features.
        save_dir (str): The directory where the video and features will be saved.
    """

    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + str(np.random.randint(10000, 99999))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(save_dir, feats_fname)
    output_mp4_path = os.path.join(save_dir, video_fname)
    np.save(output_npy_path, feats)

    if len(joints.shape) == 4:
        joints = joints[0]
    joints = joints - joints[0, 0]
    pose_generator = HybrIKJointsToRotmat()
    pose = pose_generator(joints)
    pose = np.concatenate([
        pose,
        np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)
    ], 1)
    shape = [768, 768]
    render = SMPLRender(cfg.RENDER.SMPL_MODEL_PATH)

    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    vid = []
    aroot = joints[:, 0].copy()
    aroot[:, 1] = -aroot[:, 1]
    aroot[:, 2] = -aroot[:, 2]
    params = dict(pred_shape=np.zeros([1, 10]),
                  pred_root=aroot,
                  pred_pose=pose)
    render.init_renderer([shape[0], shape[1], 3], params)
    for i in range(joints.shape[0]):
        renderImg = render.render(i)
        vid.append(renderImg)

    # out = np.stack(vid, axis=0)
    out_video = mp.ImageSequenceClip(vid, fps=model.fps)
    out_video.write_videofile(output_mp4_path, fps=model.fps)
    del render