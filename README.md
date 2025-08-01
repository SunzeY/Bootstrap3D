# <img src="assets/icon.jpg" style="vertical-align: -14px;" :height="40px" width="40px"> Bootstrap3D

**[Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](https://arxiv.org/abs/2406.00093v2)**
</br>
[Zeyi Sun](https://github.com/SunzeY),
[Tong Wu](https://wutong16.github.io/),
[Pan Zhang](https://panzhang0212.github.io/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Xiaoyi Dong](https://lightdxy.github.io/)
[Yuanjun Xiong](http://yjxiong.me/),
[Dahua Lin](http://dahua.site/),
[Jiaqi Wang](https://myownskyw7.github.io/)

<p align="center">
<a href="https://arxiv.org/abs/2406.00093v2"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://sunzey.github.io/Bootstrap3D/"><img src="https://img.shields.io/badge/Project-Website-red"></a>
</p>


<video src="https://github.com/SunzeY/SunzeY.github.io/blob/main/Bootstrap3D/videos/gaussian.mp4" controls="controls">
</video>


## 📜 News
🚀 [2025/6/23] Bootstrap3D is accepted by ICCV 2025! 

🚀 [2024/6/4] The [paper](https://arxiv.org/abs/2406.00093v2) and [project page](https://sunzey.github.io/Bootstrap3D/) are released!

## 💡 Highlights
- 🔥 A new Multi-View Diffusion model trained on high quality synthetic data and capable of generating multi-view images closely follow text prompt.
- 🔥 Denser captioned Objaverse Dataset using finetuned 3D aware MV-LLaVA powered by GPT-4V.
- 🔥 A High Quality synthetic dataset for high asethetic 3D content creation. 

## 👨‍💻 Todo
- [x] Training code of MV-Diffusion model based on PixArt.
- [x] Release of MV-PixArt-alpha.
- [x] BS-Objaverse Dataset cart launched on huggingface.
- [x] MV-LLaVA model and web demo.
- [x] Paper and project page.

## ⚡ Quick Start

### inference
Install `diffuser` with [PixArt](https://github.com/PixArt-alpha/PixArt-alpha) supported.

```python
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
```

### Training
To reproduce our result:
1. use `prompt_gen/prompt_gen.py` ask GPT-4 to generate arbitrary number of prompts
2. use Pixart-Alpha generate image based on prompts
3. use `prompt_gen/gpt_quality_check.py` to generate quality check based on GPT-4V
4. use `prompt_gen/instructions.py` to generate instructions to prompt-tune LLaVA
5. clone code of ShareGPT4V, prepare their training environment and use generated instructions to finetune MV-LLaVA (detailed in next section).
6. generate more data based on MV-LLaVA and formate data into Pixart-Alpha formate
7. clone [Pixart-Alpha](https://github.com/PixArt-alpha/PixArt-alpha) repo and prepare their environments, put `train/PixArt_xl2_img512_internal_for_3d_sample_training_long.py` in `config` folder,  `sup_file/train/train_tri.py` in `train_script` folder, `sup_file/train/train_mv_pixart_512.sh` in `.` and use a slurm supported cluster to launch the script.


## MV-LLaVA

### 📜 News
MV-LLaVA is trained on 30K GPT-4V generated instructive conversation pairs, enable LLaVA to process multi-view images rendered from 3D content, chat about it and generate dense descriptive captions or provide quality estimation.

It's 7B model is available on [huggingface](https://huggingface.co/Zery/MV-LLaVA-7B)

We use this model to provide quality estimation on Objaverse and rewrite dense descriptive captions, We call this caption dataset BS-Objaverse(**B**oot**S**trap-Objaverse), it is now available on [huggingface](https://huggingface.co/datasets/Zery/BS-Objaverse).

We also use this model to process synthetic multi-view images generated by [SV3D](https://huggingface.co/stabilityai/sv3d) and [Zero123++](https://github.com/SUDO-AI-3D/zero123plus).

### 🛠️ Usage
#### Installation (Infer only)
Our MV-LLaVA is based on [ShareGPT-4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), thanks for their awesome work!
You can clone our repo and `cd MV_LLaVA && pip install -e .` to install `share4v` package.

- launch our demo through `python app.py`
- batch inference your multi-view images using batch scripts in `tools/`

#### Installation (Training)

**training demo**
clone our repo and `cd MV_LLaVA && pip install -e .` to install `share4v` package.
first use `bash scripts/slurm_pretrain_7b_mv.sh` to align CLIP with LLama, than run `bash scripts/slrum_finetune_7b_mv.sh` to do instruct tuning.

we have uploaded a demo objaverse multi-view data (10 images only) in `data/obj_demo`, its json for pretraining and instruct tuning are available in `data/demo_obj_pretrain.json` and  `data/demo_obj_instruct.json`. You can generate your own data following the same format. It's worth noticing that pretraining data only support single-turn conversation.

You can overlook the modification [here](https://github.com/SunzeY/Bootstrap3D/commit/0a3d99de63d0d8fa323b0336f40487cbd104b33d) to MV-LLaVA's modification based on Share4V. During your own special usage, you only need to focus on these lines of code.

If you only need to change training data, you can focus on line of codes with `modify` tag (search this tag in your IDE).

**full data preparation (Objaverse)**
1. download full [cap3D dataset](https://huggingface.co/datasets/tiange/Cap3D) of objaverse rendered images.
2. download BS-Objaverse dataset GPT-4V generated annotations [obj_descript_gpt_10k.json](https://huggingface.co/datasets/Zery/BS-Objaverse), convert its into the similar format as demo did.
3. prepare [share4v dataset](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) (optional to mitigate overfitting)




## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@misc{sun2024bootstrap3dimprovingmultiviewdiffusion,
      title={Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data}, 
      author={Zeyi Sun and Tong Wu and Pan Zhang and Yuhang Zang and Xiaoyi Dong and Yuanjun Xiong and Dahua Lin and Jiaqi Wang},
      year={2024},
      eprint={2406.00093},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.00093}, 
}
```
