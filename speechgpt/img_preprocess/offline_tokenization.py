import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from seed.seed_tokenizer import SeedLlamaTokenizer
from seed.transforms import get_transform

from torch.multiprocessing import Lock, Pool, Process, set_start_method
from torchvision.datasets import ImageFolder

class LLaVAImagePretrainDataset(ImageFolder):
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path


def main():
    with open("/mnt/hdd4/xiejunlin/datasets/llava/train_json/blip_laion_cc_sbu_558k.json", "r") as fp:
        data = json.load(fp)

    result = list()

    transform = get_transform("clip", image_size=224, keep_ratio=False)
    tokenizer = SeedLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path="AILab-CVC/seed-tokenizer-2",
        encoder_url="https://hf-mirror.com/AILab-CVC/seed-tokenizer-2/resolve/main/seed_quantizer.pt",
        diffusion_path="stabilityai/stable-diffusion-2-1-unclip",
        load_diffusion=True,
        fp16=True,
    )

    tokenizer.to("cuda")
    tokenizer.pad_token = tokenizer.unk_token

    print("Init Done")
    
    # loading image folder
    dataset = LLaVAImagePretrainDataset(root="/mnt/hdd4/xiejunlin/datasets/llava/llava_image_pretrain", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)

    with open("./image_tokens.txt", "w") as f:
        for sample, path in tqdm(dataloader):
            
            image_tokens = tokenizer.encode_image(image_torch=sample.cuda())
            image_tokens = image_tokens.detach().cpu().numpy()

            for token, p in zip(image_tokens, path):
                text = f"{p}," + "".join([f"<IMG_TOKEN:{t}>" for t in token]) + "\n"
                f.write(text)

    with open("/home/xiejunlin/data/datasets/llava/train_json/blip_laion_cc_sbu_558k.json", "r") as fp:
        data = json.load(fp)
    
    with open("./image_tokens.txt", "r") as f:
        tokens = f.readlines()
        tokens = [token.strip() for token in tokens]

    tokens_list = [tuple(token.split(",")) for token in tokens]
    tokens_list = sorted(tokens_list, key=lambda x: x[0])
    data_list = sorted(data, key=lambda x: x['image'])

    with open("./llava_v1_5_665k_token.txt", "w") as f:
        for t, d in tqdm(zip(tokens_list, data_list)):
            assert d['image'] in t[0]
            text: str = (
                "Human: " 
                + d["conversations"][0]["value"]
                + " </s>"
                + " Assistant: "
                + d["conversations"][1]["value"]
                + " </s>"
                + "\n"
            )
            text = text.replace("<image>", t[1])
            f.write(text)


if __name__ == "__main__":
    main()
