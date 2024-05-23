import argparse
import torch

from share4v.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from share4v.conversation import conv_templates, SeparatorStyle
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init
from share4v.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
import math
from tqdm import tqdm

import io
import sys
sys.path.insert(0, '/mnt/petrelfs/sunzeyi/petrel-oss-python-sdk/')
from petrel_client.client import Client
from PIL import Image
import cv2
import numpy as np
import math

conf_path = '~/petreloss_my.conf'
client = Client(conf_path)

num_res_per_pack = 1000

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

gt_answers = None

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    anns = json.load(open("path_to_image_list.json", 'r'))
    print(len(anns))
    anns = get_chunk(anns, 256, args.split_index)
    print(len(anns))
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    print(model_name)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    elif "share" in model_name.lower():
        conv_mode = "share4v_v1"
    else:
        conv_mode = "share4v_v1"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    gen_res = []

    turn_mesg = [
            "<image><image><image><image>\nGenerate a short caption of the following multi-view image.",
            "What do you think about the overall quality of view consistency of three synthesized novel views? Choosing from \'poor\', \'relatively poor\', \'boardline\', \'relatively good\', \'good\', \'perfect\'.",
        ]
    turn_key = ["caption", "score"]
    score_dict = {"poor": 1, 'relatively poor': 2, 'boardline': 3, 'relatively good': 4, 'good': 5, 'perfect': 6, 'Good': 5}

    for ann in tqdm(anns):
        turn = 0
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        f = io.BytesIO(client.get(ann['image_pth']))
        image = Image.open(f)

        assert image.size[0] == 640 # for zeropp, each image is 320x320.
        v1, v2, v3, v4 = image.crop((0, 0, 320, 320)), image.crop((320, 0, 640, 320)), image.crop((0, 320, 320, 640)), image.crop((320, 320, 640, 640))
        images = [v1, v3, v4, v2]
        # Similar operation in model_worker.py
        image_tensor = process_images(images, image_processor, model.config)

        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        while True:
            if turn == 2:
                # print("exit...")
                break
            try:
                inp = turn_mesg[turn]
            except EOFError:
                inp = ""
            # print(f"{roles[1]}: ", end="") 
            # if images is not None:
            #     # already add tokens
            #     # first message
            #     # if model.config.mm_use_im_start_end:
            #     #     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            #     # else:
            #     #     inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            #     conv.append_message(conv.roles[0], inp)
            #     image = None
            # else:
            #     # later messages
            #     conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(
                        dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    # streamer=streamer,
                    use_cache=True)
            # import pdb
            # pdb.set_trace()
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(
                    f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            # outputs = tokenizer.decode(output_ids[0]).strip()
            conv.messages[-1][-1] = outputs
            
            if turn == 1:
                if outputs not in score_dict.keys():
                    print(outputs)
                    ann[turn_key[turn]] = 1
                else:
                    ann[turn_key[turn]] = score_dict[outputs]
            else:
                ann[turn_key[turn]] = outputs
            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
            
            turn += 1
        gen_res.append(ann)
        # print(ann)
    json.dump(gen_res, open(f"mv_gen_zeropp_sigma_38w/{args.split_index}.json", 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Zery/MV-LLaVA-7B")
    parser.add_argument("--model-name", type=str, default="share4v-7b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split_index", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
