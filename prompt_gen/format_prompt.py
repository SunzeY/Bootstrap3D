import json
import os
import shutil
import json
res = json.load(open("gpt/cache_result_prompt_new_500k.json", 'r'))
prompts = [] 
for item in res:
    try:
        content = item["choices"][0]["message"]["content"]
        sep = content.replace('\n', '')
        sep = json.loads(sep)
        prompts.extend(sep)
    except:
        try:
            content = item["choices"][0]["message"]["content"]
            sep = content.split("\n")
            sep_wo_number = [x.split(".")[1] for x in sep if len(x.split(".")) > 1]
            sep = [x.replace(" \"", "").replace("\"", "")for x in sep_wo_number]
            prompts.extend(sep)
        except:
            print(content)
print(len(prompts))
json.dump(prompts, open("gpt/prompts_500k_new.json", 'w'), indent=4)
