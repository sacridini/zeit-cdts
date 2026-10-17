"""FC-Siam-diff (Daudt, Le Saux & Boulch 2018): a U-Net whose encoder, shared by both
dates, passes the absolute difference of the two dates' features at every scale to the
decoder. Written from the paper; it gives the outputs of the authors' ``SiamUnet_diff``
with the same weights (the decoder's 3x3 transposed convolutions of stride 1 are plain
convolutions here, with their kernels flipped)."""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# channels of the convolutions of each stage of the encoder; the decoder mirrors them
_STAGES = ((16, 16), (32, 32), (64, 64, 64), (128, 128, 128))


def _block(in_c: int, out_c: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c),
                         nn.ReLU(inplace=True), nn.Dropout2d(dropout))


def _stage(in_c: int, channels, dropout: float) -> nn.Sequential:
    layers = []
    for out_c in channels:
        layers.append(_block(in_c, out_c, dropout))
        in_c = out_c
    return nn.Sequential(*layers)


class SiameseChangeDetector(nn.Module):
    """Change between two co-registered images, ``(batch, bands, H, W)`` each, as per-pixel
    logits ``(batch, num_classes, H, W)``. The image halves four times: ``H`` and ``W`` are
    16 or more (any size: the decoder pads an odd size back, as the authors' code does)."""

    def __init__(self, in_channels: int, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.ModuleList()
        c = in_channels
        for channels in _STAGES:
            self.encoder.append(_stage(c, channels, dropout))
            c = channels[-1]
        # decoder, deepest stage first: upsample, concatenate the difference of the
        # encoder's features at that scale, then the encoder's convolutions in reverse
        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(_STAGES))):
            channels = list(reversed(_STAGES[i]))
            self.up.append(nn.ConvTranspose2d(c, c, kernel_size=3, stride=2, padding=1, output_padding=1))
            # the last convolution goes to the channels of the stage above; the first
            # stage's last convolution is the classifier
            outs = channels[1:] + [_STAGES[i - 1][-1]] if i else channels[1:]
            self.decoder.append(_stage(c + channels[0], outs, dropout))
            c = outs[-1]
        self.classifier = nn.Conv2d(c, num_classes, kernel_size=3, padding=1)

    def forward_once(self, x: torch.Tensor) -> List[torch.Tensor]:
        """The encoder's features of one image at each scale (full, 1/2, 1/4, 1/8), and
        the bottleneck (1/16) last."""
        feats = []
        for stage in self.encoder:
            x = stage(x)
            feats.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return feats + [x]

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor) -> torch.Tensor:
        # both dates through the shared encoder in one pass (the authors' code runs it once per
        # date): the same in prediction, and in training batch normalization takes the
        # statistics of both dates together, which trains as well and faster
        n = x_t0.shape[0]
        both = self.forward_once(torch.cat([x_t0, x_t1]))
        f0, f1 = [f[:n] for f in both], [f[n:] for f in both]
        x = f1[-1]  # as in the authors' code, the decoder starts from the second date's bottleneck
        for up, stage, a, b in zip(self.up, self.decoder, reversed(f0[:-1]), reversed(f1[:-1])):
            x = up(x)
            dh, dw = a.shape[2] - x.shape[2], a.shape[3] - x.shape[3]
            if dh or dw:
                x = F.pad(x, (0, dw, 0, dh), mode="replicate")
            x = stage(torch.cat([x, torch.abs(a - b)], 1))
        return self.classifier(x)
