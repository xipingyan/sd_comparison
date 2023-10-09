import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

from PIL import Image

model_id="openai/clip-vit-base-patch16"
model_id="models/clip-vit-base-patch16"
clip_score_fn = partial(clip_score, model_name_or_path=model_id)

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def np2image(np_buf):
    im = Image.fromarray((np_buf*255).astype("uint8"))
    return im
