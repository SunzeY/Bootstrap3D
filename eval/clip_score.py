import torch
import clip
from PIL import Image
import json

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
prompts = json.load(open("prompts.json", 'r'))

with torch.no_grad():
    text = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(text)
    mean_r = 0.0
    mean_s = 0.0
    for v in range(4):
        image_features = []
        for i in range(len(prompts)):
            image = preprocess(Image.open(f"test_clip/svd/110/seed_2/{v}/{i}.png")).unsqueeze(0).to(device)
            image_feature = model.encode_image(image)
            image_features.append(image_feature)
        image_features = torch.concat(image_features, dim=0)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        logits = image_features @ text_features.T
        score = logits[torch.arange(len(prompts)).cuda(), torch.arange(len(prompts)).cuda()].mean()
        acc1, acc5 = accuracy(logits, torch.arange(len(prompts)).cuda(), topk=(1, 5))
        top1 = (acc1 / len(prompts)) * 100
        top5 = (acc5 / len(prompts)) * 100 
        print(f"Top-1 accuracy of view {v}: {top1:.2f}")
        print(f"Top-5 accuracy of view {v}: {top5:.2f}")
        print(f"CLIP score of view {v}: {score}")
        
        mean_r += top1
        mean_s += score
        
print(f"mean of 4 view: r={mean_r/4:.2f}, s={mean_s/4:.3f}")
