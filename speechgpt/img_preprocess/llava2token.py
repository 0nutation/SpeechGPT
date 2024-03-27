import json
import os

import torch.distributed as dist
import torch.multiprocessing as mp

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from seed.seed_tokenizer import SeedLlamaTokenizer
from seed.transforms import get_transform
from torch.multiprocessing import Lock, Pool, Process, set_start_method


def main():
    with open("/share/home/tj24037/song/datasets/llava/train_json/blip_laion_cc_sbu_558k.json", "r") as fp:
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

    for sample in tqdm(data):
        img = sample["image"]
        # modify path
        img = os.path.join("", img)
        img = Image.open(img).convert("RGB")
        image_tensor = transform(img).to("cuda")
        image_tokens = tokenizer.encode_image(image_torch=image_tensor)

        decoded = tokenizer.decode_image(image_tokens)
        print(image_tokens)
        decoded[0].save("./decoded.jpg")
        text: str = (
            "Given an image: "
            + sample["conversations"][0]["value"]
            + "\n"
            + "Answer: "
            + sample["conversations"][1]["value"]
        )
        text.replace("<image>", image_tokens)
        result.append(text)
        exit()


if __name__ == "__main__":
    main()
