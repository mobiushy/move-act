from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class RunConfig:
    img_path: str = './data/0012.jpg'
    cond_path: str = './condition/0012.json'

    attn_steps: int = 7

    up_ft_indices: List[int] = field(default_factory=lambda: [2])
    ensemble_size: int = 1

    inv_prompt: str = "A photo of cat"
    inv_token_idx: int = 4
    edit_prompt: str = "A running cat"

    seeds: List[int] = field(default_factory=lambda: [42])
    latents_path: Path = Path('./latents_forward')
    output_path: Path = Path('./outputs')
    n_inference_steps: int = 49
    guidance_scale: float = 7.5

    transfer_step: int = 35
    scale_factor: int = 150
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    update_steps: int = 50
    dilate_kernel_size = 15

    sd_2_1: bool = False

    bbox: List[int] = field(default_factory=lambda: [36, 186, 246, 416])
    color: List[str] = field(default_factory=lambda: ['blue', 'red', 'purple', 'orange', 'green', 'yellow', 'black'])
    P: float = 0.3

    dataset_path: Path = Path('./datasets/mydataset')
    eval_output_path: Path = Path('./outputs/eval')

    def __post_init__(self):
        self.latents_path.mkdir(exist_ok=True, parents=True)
        self.output_path.mkdir(exist_ok=True, parents=True)
    