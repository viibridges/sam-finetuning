# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, PromptEncoder, Sam, TwoWayTransformer

from .modeling import MaskDecoder
# from .custom import MaskDecoder

from .custom import MaskEncoder


def build_sam_vit_h(image_size=1024, checkpoint=None, val=False, device='cpu'):
    return _build_sam(
        image_size=image_size,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        val=val,
        device='cpu'
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size=1024, checkpoint=None, val=False, device='cpu'):
    return _build_sam(
        image_size=image_size,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        val=val,
        device='cpu'
    )


def build_sam_vit_b(image_size=1024, checkpoint=None, val=False, device='cpu'):
    return _build_sam(
        image_size=image_size,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        val=val,
        device='cpu'
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    image_size,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    val=False,
    device='cpu'
):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        mask_encoder=MaskEncoder(transformer_dim=prompt_embed_dim),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,  # allow 3 ambiguous mask outputs
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    if val:
        print("SAM in eval mode")
        sam.eval()
    else:
        print("SAM in train mode")
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device(device))
        if val:
            sam.load_state_dict(state_dict)
        elif image_size == 1024:
            new_state_dict = dict()
            for name, weights in state_dict.items():
                if name.startswith('mask_decoder'):
                    print("Skip unmatched weights:", name)
                    continue
                else:
                    new_state_dict[name] = weights
            sam.load_state_dict(new_state_dict, strict=False)
        else:
            new_state_dict = dict()
            for name, weights in state_dict.items():
                if name.startswith('image_encoder') and 'pos_' in name:
                    print("Skip unmatched weights:", name)
                    continue
                else:
                    new_state_dict[name] = weights
            sam.load_state_dict(new_state_dict, strict=False)
    return sam
