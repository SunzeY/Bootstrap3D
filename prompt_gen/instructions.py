cap_prompts = ["<image><image><image><image>\nWhat is this multi-view photo about? generate a short caption for me.",
           "<image><image><image><image>\nGenerate a short caption of the following multi-view image.",
           "<image><image><image><image>\nCan you describe the main features of this multi-view image for me by a short caption?"]

reason_prompts = ["How about the view consistency of this synthesized multi-view image?",
           "Do some comments about the view consistency of this synthesized multi-view image.",
           "What do you think about the view consistency of this synthesized multi-view image?"]

question_quality = "What do you think about the overall quality of view consistency of three synthesized novel views? Choosing from \'poor\', \'relatively poor\', \'boardline\', \'relatively good\', \'good\', \'perfect\'."

quality_list = ['poor', 'relatively poor', 'boardline', 'relatively good', 'good', 'perfect']

import json
import os
from tqdm import tqdm

gen_data = []
gen_data_cap = []
for i, line in tqdm(enumerate(json.load(open('xxx.json', 'r')))):
    gen_data.append(
        dict(
            id=f"sv3d_{i}",
            image=line['image_pth'],
            conversations=[
                {'from': 'human', 'value': cap_prompts[i%3]},
                {'from': 'gpt', 'value': line["caption"]},
                {'from': 'human', 'value': reason_prompts[i%3]},
                {'from': 'gpt', 'value': line["reason"]},
                {'from': 'human', 'value': question_quality},
                {'from': 'gpt', 'value': quality_list[line["score"]-1]},
            ],
        )
    )
    gen_data_cap.append(
        dict(
            id=f"sv3d_{i}",
            image=line['image_pth'],
            conversations=[
                {'from': 'human', 'value': cap_prompts[i%3]},
                {'from': 'gpt', 'value': line["caption"]},
            ],
        )
    )
print(len(gen_data))

json.dump(gen_data_cap, open("gpt/xxx.json", 'w'), indent=4)
json.dump(gen_data, open("gpt/xxx.json", 'w'), indent=4)
