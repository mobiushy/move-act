import os
import json


dic = {
    "cat": 15,
    "bear": 21,
    "dog": 16,
    "horse": 17
}

cond_dir = "./datasets/mydataset/condition"
txt_dir = "./datasets/eval/labels/val"

for file in os.listdir(cond_dir):
    cond_path = os.path.join(cond_dir, file)

    with open(cond_path, 'r', encoding='utf8') as fp:
        cond = json.load(fp)
    
    fname = cond["fname"].split('.')[0]
    fname = fname + '.txt'

    obj = cond["inv_prompt"].split(' ')[-1]

    xmin, ymin, xmax, ymax = cond["bbox"]

    xcent = (xmin + xmax) / 2
    ycent = (ymin + ymax) / 2
    xlen = xmax - xmin
    ylen = ymax - ymin

    xcent = xcent / 512
    ycent = ycent / 512
    xlen = xlen / 512
    ylen = ylen / 512

    with open(os.path.join(txt_dir, fname), 'a') as f:
        f.write(f"{dic[obj]} {xcent} {ycent} {xlen} {ylen}")
