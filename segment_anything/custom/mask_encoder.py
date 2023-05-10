#
# Generate low-res mask to feed to prompt-encoder from image embeddings
#
from torch import nn

from typing import Type
from ..modeling.common import LayerNorm2d

class MaskEncoder(nn.Module):
    def __init__(self,
                transformer_dim,
                activation: Type[nn.Module] = nn.GELU,
                *args, **kwargs
        ) -> None:
        super().__init__()
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim // 8, 1, kernel_size=1, stride=1),
        )

    def forward(self, image_embeddings):
        return self.deconvs(image_embeddings)
