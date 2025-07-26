import base64
import requests
import json
import os
from tqdm import tqdm
import time
import argparse
from openai import OpenAI

f = open("prompt_to_prompt.txt", 'r')
prompt = f.readlines()
prompt = "".join(prompt).replace('  ', '')

# OpenAI API Key
api_key = "Your api key" # 100

client = OpenAI(api_key=api_key,
                base_url=api_base)

def process_image(image_path):
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
            ]
            }
        ],
        "max_tokens": 3000
    }

    response = client.chat.completions.create(**payload)

    # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response
    # print(response.json())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    local_rank = args.local_rank
    os.makedirs(f"gpt/", exist_ok=True)
    args = parser.parse_args()
    if not os.path.exists(f"gpt/cache_result_prompt_new_500k.json"):
        json.dump([], open(f"gpt/cache_result_prompt_new_500k.json", 'w'), indent=4)
    data = json.load(open(f"gpt/cache_result_prompt_new_500k.json", 'r'))
    for i in range(10000):
        try:
            response = process_image("NONE")
            print(response.json())
        except:
            print("failed!")
            time.sleep(2)
            continue
        # import pdb
        # pdb.set_trace()
        data.append(json.loads(response.json()))
        json.dump(data, open(f"gpt/cache_result_prompt_new_500k.json", 'w'), indent=4)
