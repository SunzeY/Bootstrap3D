import os
from diffusers import PixArtAlphaPipeline
import torch
from diffusers import Transformer2DModel
import json
import matplotlib.pyplot as plt
import numpy as np
import textwrap

pip_dict = {512: "PixArt-alpha/PixArt-XL-2-512x512",
            1024: "PixArt-alpha/PixArt-XL-2-1024-MS"}

from PIL import Image

transformer = Transformer2DModel.from_pretrained(pretrained_model_name_or_path="Zery/MVPixArt-XL-2-512x512_sv3d", torch_dtype=torch.float16)
pipe = PixArtAlphaPipeline.from_pretrained(pip_dict[resolution], torch_dtype=torch.float16, transformer=transformer)
pipe = pipe.to("cuda")
typ = "sv3d"

prompt = "a cute puppy."

prompt_cad = f"[Four images from DIFFERENT views of a single-object with CAD style] " + prompt

# Generate images for each style
image_cad = pipe(prompt=prompt_cad).images[0]
# Save individual images (optional)
image_cad.save(f"puppy.jpg")
