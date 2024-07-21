
from typing import List, Optional
import os
import cv2
import torch
import numpy as np
from diffusers.utils import logging

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from PIL import Image
from tqdm import tqdm
from utils.vis_utils import show_cross_attention, show_mask
import torchvision.transforms as transforms
from utils.ptp_utils import register_time, load_source_latents_t
from utils.dift_sd import SDFeaturizer
from utils.ptp_utils import register_cross_attention_control_efficient


logger = logging.get_logger(__name__)

class MnAPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds



    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
    
    def load_img(self, image_path):
        image_pil = transforms.Resize(512)(Image.open(image_path).convert("RGB"))
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        image = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image
    
    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs
    

    @torch.no_grad()
    def dilate_mask(self, mask, config):
        img = torch.zeros(64, 64)
        img[mask] = 1
        img = img.numpy()
        kernel = np.ones((config.dilate_kernel_size, config.dilate_kernel_size), np.uint8)
        dilate = cv2.dilate(img, kernel, 1)

        mask_edge = torch.from_numpy(dilate)
        mask_edge = (mask_edge > 0.5).to('cuda:0')
        mask_edge[mask] = False

        return mask_edge
    
    @staticmethod
    def _update_latent(latents, loss, step_size):
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond

        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, config):

        dift = SDFeaturizer()

        scale_range = np.linspace(config.scale_range[0], config.scale_range[1], config.update_steps)

        timesteps = reversed(self.scheduler.timesteps)
        register_cross_attention_control_efficient(dift.pipe, config.inv_token_idx)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps, desc="Inversion")):
                register_time(self, t.item())

                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                if i == 0:
                    _ = dift.forward(latent,
                        prompt=config.inv_prompt,
                        t=t.item(),
                        up_ft_indices=config.up_ft_indices,
                        ensemble_size=config.ensemble_size)
                    
                    dift.pipe.controller.merge_attention()
                    attn_map = dift.pipe.controller.merge_attn_map.detach().cpu()
                    dift.pipe.controller.reset()


                    mask = (attn_map >= 0.01).to('cuda:0')

                    mask_edge = self.dilate_mask(mask, config)

                    bbox = [max(round(b / (512 / 64)), 0) for b in config.bbox]
                    x1, y1, x2, y2 = bbox

                    mask_box = torch.zeros(64, 64)
                    ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=mask_box.dtype).to(mask_box.device)
                    mask_box[y1:y2, x1:x2] = ones_mask
                    mask_box = (mask_box > 0.5).to('cuda:0')
                        

                    mask_src = mask
                    mask_src[mask_box] = False
                    mask_bg = ~(mask_src + mask_box)

                        
                    mask_src = mask_src.reshape(-1)
                    mask_edge = mask_edge.reshape(-1)
                    mask_bg = mask_bg.reshape(-1)
                    mask_box = mask_box.reshape(-1)



                if i == config.transfer_step:
                    with torch.enable_grad():

                        latent = latent.clone().detach().requires_grad_(True)

                        source_ft = dift.forward(latent,
                            prompt=config.inv_prompt,
                            t=t.item(),
                            up_ft_indices=config.up_ft_indices,
                            ensemble_size=config.ensemble_size)
                        
                        source_ft = source_ft.reshape(source_ft.shape[0], -1)

                        dift.pipe.controller.reset()
                        dift.pipe.unet.zero_grad()

                        print("updating latent code...")
                        for scale in scale_range:
                                
                            ft = dift.forward(latent,
                                prompt=config.inv_prompt,
                                t=t.item(),
                                up_ft_indices=config.up_ft_indices,
                                ensemble_size=config.ensemble_size)

                            ft = ft.reshape(ft.shape[0], -1)

                            dift.pipe.controller.merge_attention()

                            attn_map = dift.pipe.controller.merge_attn_map
                            attn_map = attn_map.reshape(-1)

                            dift.pipe.controller.reset()
                            dift.pipe.unet.zero_grad()


                            k = (mask_box.sum() * config.P).long()
                            loss_attn = 1. - torch.mean((attn_map[mask_box]).topk(k)[0])
                            loss_zero = torch.mean(attn_map[~mask_box])

                                
                            ft_edge = source_ft[:, mask_edge]
                            fts_edge = ft_edge
                            while fts_edge.shape[1] < mask_src.sum():
                                fts_edge = torch.cat([fts_edge, ft_edge], dim=1)
                                
                            loss_ipt = torch.nn.SmoothL1Loss()(ft[:, mask_src], fts_edge[:, :mask_src.sum()])
                            

                            loss_bg = torch.nn.SmoothL1Loss()(ft[:, mask_bg], source_ft[:, mask_bg])


                            loss = 0.25*loss_bg + 0.25*loss_ipt + 0.5*(loss_attn+loss_zero)

                            # print('loss_bg: ', loss_bg)
                            # print('loss_ipt: ', loss_ipt)
                            # print('loss_attn: ', loss_attn)
                            # print('loss_zero: ', loss_zero)
                            # print('loss: ', loss)
                            
                            
                            latent = self._update_latent(latents=latent, loss=loss, step_size=config.scale_factor * np.sqrt(scale))

                        print("update completed!")


                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps

                torch.save(latent, os.path.join(config.latents_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(config.latents_path, f'noisy_latents_{t}.pt'))

        return latent

    @torch.no_grad()
    def edit(
            self,
            guidance_scale: float = 7.5,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            config = None
    ):

        timesteps = self.scheduler.timesteps

        with torch.autocast(device_type='cuda', dtype=torch.float32):

            for i, t in enumerate(tqdm(timesteps, desc="Editing  ")):
                
                # register timesteps
                register_time(self, t.item())

                pnp_guidance_embeds = self.get_text_embeds(config.edit_prompt, "").chunk(2)[0]
        
                # expand the latents if we are doing classifier free guidance
                source_latents = load_source_latents_t(t, config.latents_path)
                latent_model_input = torch.cat([source_latents] + ([latents] * 2))


                text_embed_input = torch.cat([pnp_guidance_embeds, prompt_embeds], dim=0)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embed_input
                ).sample

                # perform guidance
                _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                              
        return latents
