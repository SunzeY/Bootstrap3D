import base64
import requests
import json
import os
from tqdm import tqdm
import time
import argparse
from openai import OpenAI

prompt = "Assume you are a quality checker of a diffusion model. This diffusion model is trained to achieve novel view synthesis. I give this model the image in the upper-left side and it generate novel views in the rest three images(upper-right, lower-left, lower-right).  You should tell me the quality of the generated novel view images. The score ranges from 1 to 5, representing the quality of the model from low to high. The detailed evaluation criteria are as follows:\n\
1. The novel views are difficult to discern what the image supposed to be, lacking in recognizability. It has no usable value.\n\
2. The novel views are distinguishable, clearly determine what the object/scene is similar to the given ground truth image. However, there is obvious inconsistency between the novel view synthesized images and groud-truth image. There are many obvious areas of image is blurred or indicating rotation.\n\
3. The novel views are relatively good, the inconsistency between novel view synthesized images with groud-truth image is not obvious. The blurring area indicating rotation or uncerntainty is accecptable for usage.\n\
4. The novel views are pretty good, although the might be blurring areas or less resolution. the view consistancy is well maintained.\n\
5. The novel views are excellent. It is hard to tell which image from four is ground-truth and which is synthesised.\n\
\n\
You shoud give me the overall score with one score number, with reason in next line. besides the quality check, I need you to generate a long discriptive caption for the scene/object from 4 different view. focusing on the part/object relative position, color, number of objects and so on with no more than 50 words and no less than 30 words. DO NOT MENSHION MULTI-VIEW IMAGES FROM DIFFERENT PERSPECTIVE since it is a single scene/object.  you should rearrange your result in a JSON format.\n\
if all the images(include the groud-truth image) are of low quality, just output a lowest score.\n\
Here is an example for you\n\
{\n\
\"score\": 4,\n\
\"reason\": \"The novel views generated from the model are quite convincing with a high degree of consistency in terms of texture, lighting, and color when compared to the ground-truth image. There is some minor distortion in shape and perspective, but the overall quality is high, and it maintains the realism of the scene.\"\n\
\"caption\": \"A cluster of shiny five apples, ranging from deep red to sunny yellow, sits comfortably within a rustic woven basket. Their smooth, round forms are grouped closely, reflecting light and casting soft shadows that accentuate their voluminous curves and vibrant colors.\"\n\
}\n\
"

# OpenAI API Key
api_key = "Your_api_key"
api_base = "Your_api_base"

client = OpenAI(api_key=api_key,
                base_url=api_base)

def process_image(image_path):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Path to your image
    # image_path = "cap3d_exp/0aae633338c44a1b8744e056c8c39ff9.jpg"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
    }
    response = client.chat.completions.create(**payload)

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    local_rank = args.local_rank
    os.makedirs(f"gpt/", exist_ok=True)
    args = parser.parse_args()
    if not os.path.exists(f"gpt/cache_result.json"):
        json.dump({}, open(f"gpt/cache_result.json", 'w'), indent=4)
    data = json.load(open(f"gpt/cache_result.json", 'r'))
    todo = json.load(open(f"exist_0_for_test_1w.json"))
    while len(set(todo) - set(data.keys())) != 0:
        for img_pth in tqdm(set(todo) - set(data.keys())):
            try:
                response = process_image(img_pth)
                print(response.json())
                if 'error' in json.loads(response.json()).keys():
                    print("waiting for rate limit...")
                    time.sleep(5)
                    continue
            except:
                print("failed!")
                time.sleep(2)
                continue
            data[img_pth] = json.loads(response.json())
            print(img_pth)
            json.dump(data, open(f"gpt/cache_result.json", 'w'), indent=4)
