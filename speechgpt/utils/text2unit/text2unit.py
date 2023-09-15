import os
from fairseq.models.transformer import TransformerModel
import argparse
from argparse import Namespace
from fairseq.data import encoders
import json
from tqdm import tqdm
from typing import List, Optional
import torch

class Text2Unit:
    def __init__(
        self,
        checkpoint_dir="speechgpt/utils/text2unit",
        checkpoint_file="text2unit.pt",
        data_name_or_path="speechgpt/utils/text2unit/binary",
        sentencepiece_model="speechgpt/utils/text2unit/spm.model"
        ) -> None:

        self.bpe_tokenizer = encoders.build_bpe(
            Namespace(
                bpe='sentencepiece',
                sentencepiece_model=sentencepiece_model,
            )
        )
        self.t2u = TransformerModel.from_pretrained(
            checkpoint_dir,
            checkpoint_file=checkpoint_file,
            data_name_or_path=data_name_or_path,
            
        )

    @torch.no_grad()
    def forward(
        self,
        text,
        **kwargs
    ):
        encoded_text = [self.bpe_tokenizer.encode(x).strip() for x in text] if isinstance(text, list) else self.bpe_tokenizer.encode(text).strip()
        output = self.t2u.translate(encoded_text, **kwargs)
        return [self.postprocess(x) for x in output] if isinstance(output, list) else self.postprocess(output)


    def postprocess(
        self,
        input
    ):
        return '<sosp>'+"".join(input.split())+'<eosp>'


    def __call__(self, text, **kwargs):
        return self.forward(text, **kwargs)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()


    translator = Text2Unit(
        checkpoint_dir="speechgpt/utils/text2unit",
        checkpoint_file="text2unit.pt",
        data_name_or_path="speechgpt/utils/text2unit/binary",
        sentencepiece_model="speechgpt/utils/text2unit/spm.model"
    )

    gen_args = {
        "max_len_b":1000,
        "beam":5,
    }

    units = translator(args.text, **gen_args)
    print(units)

