#
# A simple mask_decoder
#
import torch
from torch import nn
from torch.nn import functional as F

from typing import Type, Tuple
from ..modeling.common import LayerNorm2d

class MaskDecoder(nn.Module):
    def __init__(self,
                transformer_dim,
                activation: Type[nn.Module] = nn.GELU,
                num_fg_classes = 2,
                *args, **kwargs
        ) -> None:
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim // 8, self.num_fg_classes, kernel_size=1, stride=1),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        original_size: Tuple[int, ...],
    ):
        x = self.deconvs(image_embeddings)
        out = self.bilinear_upscaling(x, original_size)
        return out

    def bilinear_upscaling(
        self,
        masks: torch.Tensor,
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks