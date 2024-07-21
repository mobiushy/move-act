import pyrallis
import torch
from PIL import Image
from config import RunConfig
from torchvision import transforms
from mna_pipeline import MnAPipeline
from utils import ptp_utils

import numpy as np
from utils.drawer import DashedImageDraw

import warnings
import json, os
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if config.sd_2_1:
        stable_diffusion_version = "/path/to/sd-v2.1"
        print("stable-diffusion-2-1")
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
        # stable_diffusion_version = "../stable-diffusion-v1-4"
    stable = MnAPipeline.from_pretrained(stable_diffusion_version).to(device)

    return stable


def find_img_paths(root):
    img_paths = []
    for dir in os.listdir(root):
        for file in os.listdir(os.path.join(root, dir)):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_paths.append(os.path.join(dir, file))
    return img_paths


def run_on_prompt(model: MnAPipeline,
                  config: RunConfig) -> Image.Image:

    model.scheduler.set_timesteps(config.n_inference_steps, device=model._execution_device)

    # condition
    cond = model.get_text_embeds(config.edit_prompt, "")
    print(cond[1].unsqueeze(0))    
    cond_inv = model.get_text_embeds(config.inv_prompt, "")

    # load and encode image
    image = model.load_img(config.img_path)
    latent = model.encode_imgs(image)

    # inversion
    ptp_utils.no_register(model)
    latents = model.ddim_inversion(cond_inv[1].unsqueeze(0), latent, config)

    # edit
    attn_timesteps = model.scheduler.timesteps[config.attn_steps:] if config.attn_steps >= 0 else []
    ptp_utils.register_attention_control_efficient(model, attn_timesteps)

    edit_latent = model.edit(prompt_embeds=cond, latents=latents, config=config)

    # decode and save image
    recon_image = model.decode_latents(edit_latent)
    image = transforms.ToPILImage()(recon_image[0])
    image.save('./edit.png')

    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)

    img_paths = find_img_paths(config.dataset_path)

    for img_path in tqdm(img_paths):

        sub_dir = img_path.split('/')[-2]
        img_name = img_path.split('/')[-1].split('.')[0]

        if img_name not in ['0182']:
            continue

        cond_path = os.path.join(config.dataset_path, 'condition', img_name+'.json')
        with open(cond_path, 'r', encoding='utf8') as fp:
            cond = json.load(fp)

        
        config.inv_prompt = cond["inv_prompt"]
        config.edit_prompt = cond["edit_prompt"]
        config.bbox = cond["bbox"]
        config.img_path = os.path.join(config.dataset_path, img_path)

        for seed in config.seeds:
            ptp_utils.seed_everything(seed)
            print(f"Current seed is : {seed}")
            image = run_on_prompt(model=stable,
                                config=config)

            prompt_output_path = config.eval_output_path / sub_dir
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            if os.path.isfile(prompt_output_path / f'{img_name}_{seed}.png'):
                continue

            image.save(prompt_output_path / f'{img_name}.png')

            canvas = Image.fromarray(np.zeros((image.size[0], image.size[0], 3), dtype=np.uint8) + 220)
            draw = DashedImageDraw(canvas)

            x1, y1, x2, y2 = config.bbox
            draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[2], width=5)
            canvas.save(prompt_output_path / f'{img_name}_{seed}_bbox.png')


if __name__ == '__main__':
    main()
