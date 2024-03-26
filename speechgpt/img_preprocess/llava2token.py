import json

import glob
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import argparse
import hydra
from omegaconf import OmegaConf
import os
import pickle
from typing import Optional
import transformers
from dataclasses import dataclass, field

from torch.multiprocessing import Process, set_start_method, Lock, Pool
import torch.multiprocessing as mp
import pyrootutils
import tqdm
import uuid
import json
import time

from PIL import Image

from seed.seed_tokenizer import SeedLlamaTokenizer
from seed.transforms import get_transform


@dataclass
class ConfigPathArguments:
    image_processor: Optional[str] = field(default=None, metadata={"help": "config path of image processor"})
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    data: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})


@dataclass
class ProcessArguments:
    save_dir: Optional[str] = field(
        default=None, metadata={"help": "save dictionary of result which will be written into a sequence of .tar"})
    gpus: Optional[int] = field(default=4, metadata={"help": "number of gpus to be used"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "batch size"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "number of workers to load data per device"})


parser = transformers.HfArgumentParser((ConfigPathArguments, ProcessArguments))
cfg_path, args = parser.parse_args_into_dataclasses()



def main():
    with open('./data/llava_image_.json', 'r') as fp:
        data = json.load(fp)


    result = list()

    if cfg_path.image_processor is not None:
        processor_cfg = OmegaConf.load(cfg_path.image_processor)
        processor = hydra.utils.instantiate(processor_cfg)
    else:
        processor = None


    transform = get_transform('clip', image_size=224, keep_ratio=False)

    tokenizer = SeedLlamaTokenizer.from_pretrained(pretrained_model_name_or_path='AILab-CVC/seed-tokenizer-2', encoder_url='https://hf-mirror.com/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt', diffusion_path='stabilityai/stable-diffusion-2-1-unclip', load_diffusion=True, fp16=True)

    tokenizer.to('cuda')

    tokenizer.pad_token = tokenizer.unk_token

    # data_cfg = OmegaConf.load(cfg_path.data)
    # dataset = hydra.utils.instantiate(data_cfg, tokenizer=tokenizer, image_processor=processor, image_transform=transform)

    print('Init Done')

    for sample in data:
        img = sample['image']
        # modify path
        img = 'data/' + img
        img = Image.open(img).convert('RGB')
        image_tensor = transform(img).to('cuda')
        image_tokens = tokenizer.encode_image(image_torch=image_tensor)

        decoded = tokenizer.decode_image(image_tokens)
        print(image_tokens)
        decoded[0].save('./decoded.jpg')
        text: str = 'Given an image: ' + sample['conversations'][0]['value'] + '\n' + 'Answer: ' + sample['conversations'][1]['value']
        text.replace('<image>', image_tokens)
        result.append(text)
        exit()


if __name__ == '__main__':
    main()
