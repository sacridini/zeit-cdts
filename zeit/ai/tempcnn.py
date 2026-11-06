"""TempCNN, the temporal convolutional network for pixel time series.

Written from Pelletier, Webb & Petitjean (2019), "Temporal Convolutional Neural Network for the
Classification of Satellite Image Time Series", Remote Sensing 11(5):523, and from the authors'
code (github.com/charlotte-pel/temporalCNN, GPL-3), following the reference architecture
"3 convolutions of 64 filters, 1 dense layer of 256 units". Weights trained by other
implementations (for instance the R package sits) do not load into this class.

The model takes ``(batch, bands, times)`` (``zeit.ai.train`` and ``predict`` permute the
samples' ``(batch, times, bands)`` for it) and returns logits; the softmax is left to the loss.

Training, as the authors do it, is not in the model: Adam with a learning rate of 1e-3, batches
of 32, and an L2 penalty of 1e-6 on the kernels of every convolution and dense layer (Keras
``l2(1e-6)``, a term ``1e-6 * sum(w ** 2)`` in the loss). ``zeit.ai.train`` has no per-model
settings: ask for ``batch_size=32`` and, for an L2 close to the authors', ``weight_decay=2e-6``
(Adam's weight decay adds ``wd * w`` to the gradient, and the penalty's gradient is
``2e-6 * w``; unlike the penalty, it also reaches the biases and batch normalization).
``TempCNN.l2_penalty`` gives the exact term to add to a loss of your own.
"""

from typing import Iterable

import torch
import torch.nn as nn


class TempCNN(nn.Module):
    """Convolutions along time (kernel 5, 'same' padding), each followed by batch normalization,
    ReLU and dropout, then a dense layer with the same three, and a linear layer of logits.

    Parameters
    ----------
    in_channels
        Bands (or indices) per date.
    n_times
        Dates per series; the dense layer takes ``filters * n_times`` inputs.
    num_classes
        Outputs (logits).
    filters, kernel_size, n_conv
        The convolutions: 64 filters, kernel 5, three of them (the authors' reference).
    dense_units
        Width of the hidden dense layer (256).
    dropout_rate
        Dropout of every block (0.5), element by element.
    """

    def __init__(self, in_channels: int, n_times: int, num_classes: int = 5, filters: int = 64,
                 kernel_size: int = 5, n_conv: int = 3, dense_units: int = 256,
                 dropout_rate: float = 0.5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd, for 'same' padding")
        self.n_times = n_times
        blocks = []
        c = in_channels
        for _ in range(n_conv):
            # Keras' batch normalization: momentum 0.99 (PyTorch's 0.01) and epsilon 1e-3
            blocks += [nn.Conv1d(c, filters, kernel_size, padding=kernel_size // 2),
                       nn.BatchNorm1d(filters, eps=1e-3, momentum=0.01), nn.ReLU(), nn.Dropout(dropout_rate)]
            c = filters
        self.conv = nn.Sequential(*blocks)
        self.dense = nn.Sequential(
            nn.Flatten(), nn.Linear(filters * n_times, dense_units),
            nn.BatchNorm1d(dense_units, eps=1e-3, momentum=0.01), nn.ReLU(), nn.Dropout(dropout_rate))
        self.out = nn.Linear(dense_units, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """The authors' initialization: He normal on the hidden kernels, Glorot uniform on the
        output's, zero biases."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if m is self.out:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def kernels(self) -> Iterable[torch.Tensor]:
        """The kernels the authors' L2 penalty covers: every convolution and dense layer."""
        return [m.weight for m in self.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]

    def l2_penalty(self, rate: float = 1e-6) -> torch.Tensor:
        """``rate * sum(w ** 2)`` over ``kernels()`` (the authors' ``l2(1e-6)``)."""
        return rate * sum((w ** 2).sum() for w in self.kernels())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, in_channels, n_times)``; returns ``(batch, num_classes)`` logits."""
        return self.out(self.dense(self.conv(x)))
