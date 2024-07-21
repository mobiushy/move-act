import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms

from utils import ptp_utils


def show_mask(mask, save_name):
    mask_vis = torch.zeros(64, 64)
    mask_vis[mask] = 1
    transforms.ToPILImage()(mask_vis.unsqueeze(0)).save(save_name)

def show_cross_attention(prompt: str,
                         attention_map,
                         tokenizer,
                         token_idx,
                         orig_image=None,
                         res_fname: str=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    images = []

    # show spatial attention for indices of tokens to strengthen
    
    image = attention_map
    image = show_image_relevance(image, orig_image)
    image = image.astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((256, 256)))
    image = ptp_utils.text_under_image(image, decoder(int(tokens[token_idx])))
    images.append(image)

    result = ptp_utils.view_images(np.stack(images, axis=0))
    result.save(res_fname)


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
