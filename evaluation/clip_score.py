import os
import json
from PIL import Image
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore

def find_img_paths(root):
    img_paths = []
    for file in os.listdir(root):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_paths.append(file)
    return img_paths


metric = CLIPScore(model_name_or_path="/path/to/clip-base")

root_dir = "/path/to/eval"
img_paths = find_img_paths(root_dir)
print(img_paths)

mScore = 0.

for img_path in img_paths:

    image_pil = Image.open(os.path.join(root_dir, img_path))
    image = transforms.ToTensor()(image_pil).unsqueeze(0)

    img_name = img_path.split('/')[-1].split('.')[0]

    cond_path = os.path.join('./datasets/mydataset', 'condition', img_name+'.json')
    with open(cond_path, 'r', encoding='utf8') as fp:
        cond = json.load(fp)

    text = cond["edit_prompt"]


    score = metric(image, text)    
    print(img_name, score.detach())

    mScore = mScore + score.item()

mScore = mScore / len(img_paths)
print(mScore)
