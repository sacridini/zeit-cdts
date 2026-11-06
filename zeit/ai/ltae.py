"""L-TAE, the lightweight temporal attention encoder, and LightTAE, a pixel classifier built on it.

Written from Sainte Fare Garnot & Landrieu (2020), "Lightweight Temporal Self-Attention for
Classifying Satellite Images Time Series", and from the authors' code
(github.com/VSainteuf/lightweight-temporal-attention-pytorch, MIT). Where the paper and the code
differ, the code is followed (sine and cosine positional encoding, one key layer over all the
channels, the extra layer norms). Weights trained by other implementations (for instance the R
package sits) do not load into these classes.

The authors' encoder of each date is a pixel-set encoder, which summarizes the pixels of a
parcel; zeit's samples are single pixels, so ``LightTAE`` replaces it by a per-date MLP (linear,
batch normalization and ReLU, 32 and 64 units, then a linear layer of 128 units and batch
normalization, the layout of the pixel-set encoder's two MLPs without the pooling between them).

Training, as the authors do it, is not in the model: Adam with the default learning rate of
1e-3, batches of 128, 100 epochs, a focal loss with gamma 1 (``loss="focal"`` in
``zeit.ai.train``, whose own focal gamma may differ), z-scored inputs and the best epoch by
validation mIoU. ``zeit.ai.train`` has no per-model settings: it normalizes by the 2 % and 98 %
quantiles and stops on the validation loss, for every model.
"""

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn


def positional_table(day_offsets: Sequence[float], d: int, T: int = 1000, repeat: int = 1) -> torch.Tensor:
    """The authors' sinusoidal table, ``(len(day_offsets), d * repeat)``.

    ``table[i, j] = sin(p / T ** (2 * (j // 2) / d))`` for even ``j``, ``cos`` of the same for odd
    ``j``, with ``p`` the day of date ``i`` (days since the first date); the ``d`` columns are
    repeated ``repeat`` times (one copy per head).
    """
    p = torch.as_tensor(day_offsets, dtype=torch.float32)
    j = torch.arange(d, dtype=torch.float32)
    angle = p[:, None] / torch.pow(torch.tensor(float(T)), 2 * torch.div(j, 2, rounding_mode="floor") / d)[None, :]
    table = torch.where(j[None, :] % 2 == 0, torch.sin(angle), torch.cos(angle))
    return table.repeat(1, repeat)


class _MasterQueryAttention(nn.Module):
    """Attention of the L-TAE: one learned query per head, keys from a single linear layer over
    all the channels, values the channels themselves split in ``n_head`` groups."""

    def __init__(self, n_head: int, d_k: int, d_in: int, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.n_head, self.d_k, self.d_in = n_head, d_k, d_in
        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, mean=0.0, std=math.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0.0, std=math.sqrt(2.0 / d_k))
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``z``: ``(B, T, E)``. Returns ``(B, E)`` and the weights ``(B, n_head, T)``."""
        B, T, E = z.shape
        keys = self.fc1_k(z).view(B, T, self.n_head, self.d_k)
        scores = torch.einsum("hk,bthk->bht", self.Q, keys) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)                      # over the dates
        values = z.view(B, T, self.n_head, E // self.n_head)
        out = torch.einsum("bht,bthc->bhc", self.dropout(attn), values)
        return out.reshape(B, E), attn


class LTAE(nn.Module):
    """Lightweight temporal attention encoder: a series ``(batch, times, in_channels)`` to one
    vector ``(batch, n_neurons[-1])``.

    Parameters
    ----------
    in_channels
        Channels of each date.
    day_offsets
        Day of each date, counted from the first one (``len`` is the number of dates).
    n_head, d_k
        Heads (16) and key size (8).
    n_neurons
        Widths of the MLP after the heads (256, 128); the first must be the model width.
    dropout
        Dropout after that MLP (0.2). The attention weights have a dropout of 0.1.
    d_model
        Width the channels are projected to (256, with a 1x1 convolution and a layer norm over
        channels and dates); ``None``: no projection, the width is ``in_channels``.
    T
        Temperature of the positional encoding (1000).
    init
        ``"authors"``: the code's initialization (normal weights and biases of the projection,
        Xavier normal on the linear layers, normal biases and batch normalization scales);
        ``"default"``: PyTorch's.
    """

    def __init__(self, in_channels: int, day_offsets: Sequence[float], n_head: int = 16, d_k: int = 8,
                 n_neurons: Sequence[int] = (256, 128), dropout: float = 0.2,
                 d_model: Union[int, None] = 256, T: int = 1000, init: str = "authors") -> None:
        super().__init__()
        if init not in ("authors", "default"):
            raise ValueError("init is 'authors' or 'default'")
        self.n_times = len(day_offsets)
        self.n_head = n_head
        self.d_model = d_model if d_model is not None else in_channels
        if self.d_model % n_head:
            raise ValueError(f"the width {self.d_model} is not a multiple of the {n_head} heads")
        n_neurons = list(n_neurons)
        if n_neurons[0] != self.d_model:
            raise ValueError(f"n_neurons[0] must be the model width, {self.d_model}")
        self.in_norm = nn.LayerNorm(in_channels)
        self.inconv = (nn.Sequential(nn.Conv1d(in_channels, d_model, 1), nn.LayerNorm((d_model, self.n_times)))
                       if d_model is not None else None)
        self.register_buffer("positions", positional_table(day_offsets, self.d_model // n_head, T, n_head))
        self.attention = _MasterQueryAttention(n_head, d_k, self.d_model)
        layers = []
        for a, b in zip(n_neurons[:-1], n_neurons[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(n_neurons[-1])
        if init == "authors":
            self._authors_init()

    def _authors_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0.0, 1.0)
                nn.init.normal_(m.bias, 0.0, 1.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias, 0.0, 1.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 0.0, 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """``x``: ``(batch, times, in_channels)``. Returns ``(batch, n_neurons[-1])`` (and the
        attention weights ``(batch, n_head, times)`` with ``return_attention``)."""
        if x.shape[1] != self.n_times:
            raise ValueError(f"this L-TAE takes {self.n_times} dates, got {x.shape[1]}")
        x = self.in_norm(x)
        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, attn = self.attention(x + self.positions)
        out = self.out_norm(self.dropout(self.mlp(out)))
        return (out, attn) if return_attention else out


class LightTAE(nn.Module):
    """A per-date MLP, the L-TAE and an MLP decoder: a pixel's series ``(batch, times, n_bands)``
    to logits ``(batch, n_labels)``.

    Parameters
    ----------
    n_bands, day_offsets, n_labels
        Bands per date, day of each date (since the first) and classes.
    encoder_mlp
        Widths of the per-date MLP (32, 64, 128): linear, batch normalization and ReLU for all
        but the last, which has no ReLU. The last is the L-TAE's ``in_channels``.
    decoder_mlp
        Hidden widths of the decoder (64, 32): linear, batch normalization and ReLU, then a
        linear layer of logits.
    **ltae
        Arguments of ``LTAE`` (``n_head``, ``d_k``, ``n_neurons``, ``dropout``, ``d_model``,
        ``T``, ``init``).
    """

    def __init__(self, n_bands: int, day_offsets: Sequence[float], n_labels: int,
                 encoder_mlp: Sequence[int] = (32, 64, 128), decoder_mlp: Sequence[int] = (64, 32),
                 **ltae) -> None:
        super().__init__()
        widths = [n_bands, *encoder_mlp]
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b)]
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.temporal = LTAE(widths[-1], day_offsets, **ltae)
        widths = [self.temporal.out_norm.normalized_shape[0], *decoder_mlp]
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], n_labels))
        self.decoder = nn.Sequential(*layers)
        if ltae.get("init", "authors") == "authors":
            for m in list(self.encoder) + list(self.decoder):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.normal_(m.bias, 0.0, 1.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.normal_(m.weight, 0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, times, n_bands)``; returns ``(batch, n_labels)`` logits."""
        B, T, C = x.shape
        x = self.encoder(x.reshape(B * T, C)).view(B, T, -1)     # batch normalization over B * T rows
        return self.decoder(self.temporal(x))
