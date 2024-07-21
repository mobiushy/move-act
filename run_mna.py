import json
import pyrallis
import torch
from PIL import Image
from config import RunConfig
from mna_pipeline import MnAPipeline
from utils import ptp_utils, vis_utils
from utils.drawer import DashedImageDraw
import numpy as np
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "/path/to/sd-v2.1"
        print("stable-diffusion-2-1")
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
        # stable_diffusion_version = "../stable-diffusion-v1-4"
    stable = MnAPipeline.from_pretrained(stable_diffusion_version).to(device)

    return stable

def run_on_prompt(model: MnAPipeline,
                  config: RunConfig) -> Image.Image:

    model.scheduler.set_timesteps(config.n_inference_steps, device=model._execution_device)

    # condition
    cond = model.get_text_embeds(config.edit_prompt, "")
    cond_inv = model.get_text_embeds(config.inv_prompt, "")

    # load and encode image
    image = model.load_img(config.img_path)
    latent = model.encode_imgs(image)

    # inversion
    ptp_utils.no_register(model)
    latents = model.ddim_inversion(cond_inv[1].unsqueeze(0), latent, config)

    # editing
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

    with open(config.cond_path, 'r', encoding='utf8') as fp:
        cond = json.load(fp)
        
    config.inv_prompt = cond["inv_prompt"]
    config.edit_prompt = cond["edit_prompt"]
    config.bbox = cond["bbox"]


    images = []
    for seed in config.seeds:
        ptp_utils.seed_everything(seed)
        print(f"Current seed is : {seed}")
        image = run_on_prompt(model=stable,
                              config=config)
        prompt_output_path = config.output_path / config.edit_prompt[:100]
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

        canvas = Image.fromarray(np.zeros((image.size[0], image.size[0], 3), dtype=np.uint8) + 220)
        draw = DashedImageDraw(canvas)

        
        x1, y1, x2, y2 = config.bbox
        draw.dashed_rectangle([(x1, y1), (x2, y2)], dash=(5, 5), outline=config.color[2], width=5)
        canvas.save(prompt_output_path / f'{seed}_bbox.png')

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.edit_prompt}.png')


if __name__ == '__main__':
    main()
