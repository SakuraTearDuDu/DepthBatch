# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import math
from typing import Callable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .dinov2_layers import (
    Mlp,
    MemEffAttention,
    NestedTensorBlock as Block,
    PatchEmbed,
    SwiGLUFFNFused,
)


logger = logging.getLogger("dinov2")


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for block in self:
            x = block(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )
        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer in {"swiglufused", "swiglu"}:
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            ffn_layer = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError
        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[index],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for index in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunk_size = depth // block_chunks
            for index in range(0, depth, chunk_size):
                chunked_blocks.append(
                    [nn.Identity()] * index + blocks_list[index : index + chunk_size]
                )
            self.blocks = nn.ModuleList([BlockChunk(part) for part in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, width, height):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        num_positions = self.pos_embed.shape[1] - 1
        if npatch == num_positions and width == height:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        width0 = width // self.patch_size + self.interpolate_offset
        height0 = height // self.patch_size + self.interpolate_offset
        sqrt_positions = math.sqrt(num_positions)
        scale_x, scale_y = float(width0) / sqrt_positions, float(height0) / sqrt_positions
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_positions), int(sqrt_positions), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(scale_x, scale_y),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        assert int(width0) == patch_pos_embed.shape[-2]
        assert int(height0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        batch, _channels, width, height = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, width, height)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(batch, -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output = []
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take)
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output = []
        index = 0
        total_block_len = len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for block in block_chunk[index:]:
                x = block(x)
                if index in blocks_to_take:
                    output.append(x)
                index += 1
        assert len(output) == len(blocks_to_take)
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        outputs = (
            self._get_intermediate_layers_chunked(x, n)
            if self.chunked_blocks
            else self._get_intermediate_layers_not_chunked(x, n)
        )
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            batch, _, width, height = x.shape
            outputs = [
                out.reshape(batch, width // self.patch_size, height // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        result = self.forward_features(*args, **kwargs)
        if is_training:
            return result
        return self.head(result["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def DINOv2(model_name):
    model_zoo = {"vits": vit_small, "vitb": vit_base, "vitl": vit_large, "vitg": vit_giant2}
    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
