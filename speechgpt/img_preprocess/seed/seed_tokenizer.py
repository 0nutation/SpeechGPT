import torch.nn as nn
import torch
# import math
# from torchvision import transforms
import os
# from timm.models import create_model
from typing import Any, Dict, List, Optional, Union
from transformers import LlamaTokenizer
from diffusers import DiffusionPipeline
# from torchvision.transforms.functional import pil_to_tensor

# import torch
from PIL import Image
from torchvision import transforms

# from qformer.qformer_quantizer import Blip2QformerQuantizer
# from diffusers import StableUnCLIPImg2ImgPipeline

WEIGHTS_NAME = 'seed_quantizer.pt'
DIFFUSION_NAME = 'diffusion_model'


class ImageTokenizer(nn.Module):
    def __init__(self,
                 model_path,
                 diffusion_model_path=None,
                 load_diffusion=False,
                 image_size=224,
                 device='cuda',
                 fp16=True,
                 **kwargs):
        super().__init__()
        from .qformer_quantizer import Blip2QformerQuantizer

        model = Blip2QformerQuantizer.from_pretrained(pretrained_model_path=model_path,
                                                      vit_precision='fp16' if fp16 else 'fp32',
                                                      **kwargs).eval()
        if diffusion_model_path is not None and load_diffusion:
            from .pipeline_sd_unclip_vit import StableUnCLIPImg2ImgPipeline
            # diffusion_model = DiffusionPipeline.from_pretrained(diffusion_model_path,
            #                                                     torch_dtype=torch.float16 if fp16 else torch.float32)
            diffusion_model = StableUnCLIPImg2ImgPipeline.from_pretrained(diffusion_model_path,
                                                                          torch_dtype=torch.float16 if fp16 else torch.float32)
            self.diffusion_model = diffusion_model.to(device)
        else:
            self.diffusion_model = None

        model = model.to(device)

        processor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=3),
            # transforms.Resize(image_size, interpolation=3),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        if fp16:
            model = model.half()

        shape_latents = torch.Size([1, 4, 96, 96])
        self.latents = torch.randn(shape_latents, generator=None, device=device, dtype=torch.float16, layout=torch.strided)

        shape_noise = torch.Size([1, 1024])
        self.noise = torch.randn(shape_noise, generator=None, device=device, dtype=torch.float16, layout=torch.strided)

        self.model = model
        self.processor = processor
        self.device = device
        self.fp16 = fp16

    def __len__(self):
        return self.model.n_embed

    def encode(self, image_torch):
        '''Convert a batch of img to code
        Args:
            model: The tokenizer model.
            img: [b, c, h, w]
        '''
        if len(image_torch.shape) == 3:
            image_torch = image_torch.unsqueeze(0)

        # img = image_torch.to(self.device)
        img = image_torch
        if self.fp16:
            img = img.half()
        with torch.no_grad():
            id, _ = self.model.get_codebook_indices(img)
        return id.view(img.shape[0], -1)

    def decode(self, indices, negative_indices=None, guidance_scale=10, num_inference_steps=20):
        image_embeds = self.model.get_codebook_entry(indices)
        # image = self.diffusion_model(image_embeds=image_embed,
        #                              noise_level=0,
        #                              num_inference_steps=20,
        #                              latents=self.latents,
        #                              noise=self.noise).images
        if negative_indices is not None:
            assert indices.shape == negative_indices.shape, 'Negative indices must have the same shape with indices'
            negative_image_embeds = self.model.get_codebook_entry(negative_indices)
        else:
            negative_image_embeds = None

        image = self.diffusion_model(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            guidance_scale=guidance_scale,
            noise_level=0,
            num_inference_steps=num_inference_steps,
            latents=self.latents,
        ).images
        return image


class SeedLlamaTokenizer(LlamaTokenizer):
    def __init__(self,
                 vocab_file,
                 unk_token="<unk>",
                 bos_token="<s>",
                 eos_token="</s>",
                 pad_token=None,
                 sp_model_kwargs: Optional[Dict[str, Any]] = None,
                 add_bos_token=True,
                 add_eos_token=False,
                 clean_up_tokenization_spaces=False,
                 device='cuda',
                 fp16=True,
                 load_diffusion=False,
                 encoder_url=None,
                 diffusion_path=None,
                 **kwargs):
        super().__init__(vocab_file, unk_token, bos_token, eos_token, pad_token, sp_model_kwargs, add_bos_token, add_eos_token,
                         clean_up_tokenization_spaces, **kwargs)
        self.device = device
        self.fp16 = fp16
        self.pad_token = self.unk_token
        self.load_diffusion = load_diffusion
        self.encoder_url = encoder_url
        self.diffusion_path = diffusion_path

        self.load_image_tokenizer()

    def load_image_tokenizer(self):
        if not hasattr(self, '_image_tokenizer'):
            if self.encoder_url is not None:
                model_path = self.encoder_url
            else:
                assert hasattr(self, 'name_or_path') and os.path.exists(self.name_or_path)
                model_path = os.path.join(self.name_or_path, WEIGHTS_NAME)
            # diffusion_model_path = os.path.join(self.name_or_path, DIFFUSION_NAME)
            # diffusion_model_path = 'stabilityai/stable-diffusion-2-1-unclip'
            self._image_tokenizer = ImageTokenizer(model_path=model_path,
                                                   diffusion_model_path=self.diffusion_path,
                                                   load_diffusion=self.load_diffusion,
                                                   device=self.device,
                                                   fp16=self.fp16)

    @property
    def image_tokenizer(self):
        if not hasattr(self, '_image_tokenizer'):
            if self.encoder_url is not None:
                model_path = self.encoder_url
            else:
                assert hasattr(self, 'name_or_path') and os.path.exists(self.name_or_path)
                model_path = os.path.join(self.name_or_path, WEIGHTS_NAME)
            # diffusion_model_path = os.path.join(self.name_or_path, DIFFUSION_NAME)
            # diffusion_model_path = 'stabilityai/stable-diffusion-2-1-unclip'
            self._image_tokenizer = ImageTokenizer(model_path=model_path,
                                                   diffusion_model_path=self.diffusion_path,
                                                   load_diffusion=self.load_diffusion,
                                                   device=self.device,
                                                   fp16=self.fp16)
        return self._image_tokenizer

    @property
    def num_image_tokens(self):
        return 8192  # self.image_tokenizer.num_tokens # allow not load

    def to(self, device):
        self.device = device
        if hasattr(self, '_image_tokenizer'):
            self._image_tokenizer.to(device=device)

    def encode_image(
        self,
        image_path=None,
        image_pil=None,
        image_torch=None,
        image_size: int = 224,
    ):
        assert (image_path is None) + (image_pil is None) + (image_torch is None) == 2

        # need_norm_to_1 = False
        if image_path is not None:
            image_pil = Image.open(image_path).convert('RGB')

        if image_pil is not None:
            image_torch = self.image_tokenizer.processor(image_pil)

            image_torch = image_torch.to(self.device)
        return self.image_tokenizer.encode(image_torch)

    def decode_image(self, indices, negative_indices=None, guidance_scale=10):
        indices = indices.to(self.device)
        if negative_indices is not None:
            negative_indices = negative_indices.to(self.device)
        image = self.image_tokenizer.decode(
            indices,
            negative_indices=negative_indices,
            guidance_scale=guidance_scale,
        )
        return image
