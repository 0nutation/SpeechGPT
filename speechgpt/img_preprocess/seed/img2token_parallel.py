# 导入必要的库和模块
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import argparse
import hydra
from omegaconf import OmegaConf
import os
import lmdb
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

import webdataset as wds
# import multiprocessing

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


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
    print(cfg_path, args)
    os.makedirs(args.save_dir, exist_ok=True)

    # lock = mp.Lock()
    # torch.multiprocessing.spawn(run_worker, args=(lock, ), nprocs=args.gpus, join=True)

    # with Pool(processes=args.gpus) as pool:
    #     pool.map(run_worker, [(i, lock) for i in range(args.gpus)])

    children = []
    for i in range(args.gpus):
        subproc = mp.Process(target=run_worker, args=(i, ))
        children.append(subproc)
        subproc.start()

    for i in range(args.gpus):
        children[i].join()


# 定义worker函数
def run_worker(gpu):
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6668', world_size=args.gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    sub_save_dir = os.path.join(args.save_dir, 'part-{:04d}'.format(gpu))
    os.makedirs(sub_save_dir, exist_ok=True)

    save_pattern = sub_save_dir + "/%07d.tar"
    if cfg_path.image_processor is not None:
        processor_cfg = OmegaConf.load(cfg_path.image_processor)
        processor = hydra.utils.instantiate(processor_cfg)
    else:
        processor = None

    if cfg_path.image_transform is not None:
        transform_cfg = OmegaConf.load(cfg_path.image_transform)
        transform = hydra.utils.instantiate(transform_cfg)
    else:
        transform = None

    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg, device='cuda')
    tokenizer.pad_token = tokenizer.unk_token

    data_cfg = OmegaConf.load(cfg_path.data)
    dataset = hydra.utils.instantiate(data_cfg, tokenizer=tokenizer, image_processor=processor, image_transform=transform)

    print('Init Done')

    # 初始化DistributedSampler和DataLoader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print(dataset)
    # 在每个GPU设备上运行模型
    with wds.ShardWriter(save_pattern, maxcount=10000) as sink:
        with torch.no_grad():
            time1 = time.time()
            for batch in tqdm.tqdm(data_loader):
                time2 = time.time()
                if gpu == 0:
                    print('time: ', time2 - time1)
                time1 = time2
                image_tensor = batch['pixel_values'].cuda()
                texts = batch['text']
                metadatas = batch['metadata']
                # key_strs = batch['__key__']
                # print(image_tensor.shape)
                # image_ids = tokenizer.encode_image(image_torch=image_tensor, compress_rate=args.compress_rate)
                image_ids = tokenizer.encode_image(image_torch=image_tensor)
                # print(image_ids.shape)

                for image_id, metadata, text in zip(image_ids, metadatas, texts):
                    key_str = uuid.uuid4().hex
                    sample = {'image_ids': image_id.view(-1).cpu().tolist(), 'text': text, 'metadata': json.loads(metadata)}
                    # sink.write({'__key__': key_str, 'json': sample})
                    sink.write({'__key__': key_str, 'pkl': pickle.dumps(sample)})


if __name__ == '__main__':
    # with multiprocessing.Pool(args.gpus) as pool:
    #     pool.map(run_worker, range(args.gpus))

    # set_start_method('spawn')
    main()

# python3 src/tools/extract_image_ids_to_torchdata_parallel.py --tokenizer configs/tokenizer/seed_llama_tokenizer.yaml --image_transform configs/processer/blip_transform.yaml --data configs/data/caption_torchdata_preprocess.yaml --save_dir dataset/seed_llama/caption/unsplash_cc3m/ --batch_size 1024 --num_workers 8 --gpus 8
