"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import numpy as np
from functools import partial
from einops import rearrange

from .blip2 import Blip2Base, disabled_train
from .vit import Block
from .utils import download_cached_file, is_url

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    # def l2norm(self, t):
    #     return F.normalize(t, p = 2, dim = -1)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        bz = z.shape[0]
        z_flattened = z.view(-1, self.e_dim)
        #print('z_flattened', z_flattened.shape)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z)**2) + torch.mean((z_q - z.detach())**2)
        else:
            loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = z_q.reshape(bz, -1, z_q.shape[-1])
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class Blip2QformerQuantizer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(self,
                 vit_model="eva_clip_g",
                 img_size=224,
                 drop_path_rate=0,
                 use_grad_checkpoint=False,
                 vit_precision="fp16",
                 freeze_vit=True,
                 num_query_token=32,
                 cross_attention_freq=2,
                 embed_dim=256,
                 max_txt_len=32,
                 codebook_embed_dim=32,
                 n_embed=8192,
                 recon_s=True,
                 blocks_for_image=True,
                 decode_depth=4,
                 use_recon_s_for_image=False,
                 use_qformer_image=False,
                 image_features_dim=1024):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model, img_size, drop_path_rate, use_grad_checkpoint,
                                                                       vit_precision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            self.ln_vision.weight.requires_grad = False
            self.ln_vision.bias.requires_grad = False

        self.codebook_embed_dim = codebook_embed_dim
        self.n_embed = n_embed
        self.recon_s = recon_s
        self.blocks_for_image = blocks_for_image
        self.use_recon_s_for_image = use_recon_s_for_image
        self.depth = decode_depth
        self.image_features_dim = image_features_dim
        self.use_qformer_image = use_qformer_image

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, self.visual_encoder.num_features)

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False

        self.quantize = VectorQuantizer2(n_embed, codebook_embed_dim, beta=0.25, remap=None, sane_index_shape=False)

        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size),
            nn.Tanh(),
            nn.Linear(self.Qformer.config.hidden_size, codebook_embed_dim)  # for quantize
        )

        self.decode_task_layer = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, self.Qformer.config.hidden_size)  # for quantize
        )

        self.quantize = self.quantize.eval()
        self.quantize.training = False
        for name, param in self.named_parameters():
            if 'quantize' in name or 'encode_task_layer' in name or 'decode_task_layer' in name:
                #print('freeze params', name)
                param.requires_grad = False

        if self.recon_s:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
            self.blocks = nn.ModuleList([
                Block(dim=self.Qformer.config.hidden_size,
                      num_heads=12,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
            ])

        if self.blocks_for_image:
            self.pos_embed_image = nn.Parameter(torch.zeros(1, num_query_token, self.Qformer.config.hidden_size))
            self.blocks_image = nn.ModuleList([
                Block(dim=self.Qformer.config.hidden_size,
                      num_heads=12,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(self.depth)
            ])

        if self.use_qformer_image:
            num_reverse_token = 1
            self.Reverse_Qformer, self.reverse_tokens = self.init_Qformer(num_reverse_token, self.Qformer.config.hidden_size)

            self.Reverse_Qformer.cls = None
            self.Reverse_Qformer.bert.embeddings.word_embeddings = None
            self.Reverse_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Reverse_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.distill_image_proj = nn.Linear(self.Qformer.config.hidden_size, image_features_dim)

        else:
            self.image_down = nn.Sequential(
                nn.Linear(self.Qformer.config.hidden_size, 256, bias=False),
                nn.ReLU(),
                nn.Linear(256, 128, bias=False),
                nn.ReLU(),
                nn.Linear(128, 32, bias=False),
            )
            self.distill_image_proj = nn.Linear(num_query_token * 32, image_features_dim)

    def get_codebook_indices(self, image):
        with torch.no_grad():
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            query_output_down = self.encode_task_layer(query_output.last_hidden_state)
            quant, loss_embed, embed_ind = self.quantize(query_output_down)
            embed_ind = embed_ind.reshape(quant.shape[0], -1)

            query_output_up = self.decode_task_layer(quant)

        return embed_ind, query_output_up

    def get_codebook_entry(self, indices):
        quant_embedding = self.quantize.get_codebook_entry(indices)
        # print('quant_embedding_shape: ', quant_embedding.shape)
        # print(self.decode_task_layer)
        # exit()
        query_output_up = self.decode_task_layer(quant_embedding)

        pos_embed_image = self.pos_embed_image.repeat(query_output_up.shape[0], 1, 1)
        query_output_up_pos_image = query_output_up + pos_embed_image
        for blk in self.blocks_image:
            query_output_up_pos_image = blk(query_output_up_pos_image)
        query_output_up = query_output_up_pos_image

        if self.use_qformer_image:
            query_atts = torch.ones(query_output_up.size()[:-1], dtype=torch.long).to(query_output_up.device)
            reverse_tokens = self.reverse_tokens.expand(query_output_up.shape[0], -1, -1)
            reverse_output = self.Reverse_Qformer.bert(
                query_embeds=reverse_tokens,
                encoder_hidden_states=query_output_up,
                encoder_attention_mask=query_atts,
                return_dict=True,
            )
            reverse_output = reverse_output.last_hidden_state
            reverse_output_proj = self.distill_image_proj(reverse_output).squeeze(1)
        else:
            reverse_output = self.image_down(query_output_up)
            reverse_output = reverse_output.reshape(reverse_output.shape[0], -1)
            reverse_output_proj = self.distill_image_proj(reverse_output)

        return reverse_output_proj

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        vit_model = kwargs.get("vit_model", "eva_clip_g")
        img_size = kwargs.get("image_size", 224)
        num_query_token = kwargs.get("num_query_token", 32)
        cross_attention_freq = kwargs.get("cross_attention_freq", 2)

        drop_path_rate = kwargs.get("drop_path_rate", 0)
        use_grad_checkpoint = kwargs.get("use_grad_checkpoint", False)
        vit_precision = kwargs.get("vit_precision", "fp16")
        freeze_vit = kwargs.get("freeze_vit", True)

        max_txt_len = kwargs.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )

        if pretrained_model_path.startswith('http'):
            print('start download seed model...')
            cached_file = download_cached_file(pretrained_model_path, check_hash=False, progress=True)
            print(cached_file)
            ckpt = torch.load(cached_file, map_location="cpu")
        else:
            ckpt = torch.load(pretrained_model_path, map_location="cpu")
        missing, unexcepted = model.load_state_dict(ckpt, strict=False)
        print('missing keys: ', len(missing), 'unexpected keys:', len(unexcepted))
        return model
