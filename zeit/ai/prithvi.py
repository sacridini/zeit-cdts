"""Prithvi-EO-2.0 (Szwarcman et al. 2024): a vision transformer pretrained by NASA and IBM as
a masked autoencoder on 4.2 million time series of HLS (Harmonized Landsat Sentinel-2)
images. Ported from the authors' ``prithvi_mae.py`` (Apache-2.0); with their weights its
encoder gives theirs, without ``transformers``, ``timm`` or remote code.

``Prithvi`` reads the six HLS bands of a zeit cube in their own units, NaN where missing,
standardizes them as Prithvi was trained, and adds a decoder that brings the encoder's
16 x 16 tokens back to the pixels and, with the pixels' own bands, to a class per pixel.
"""
import math
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Prithvi's six bands (HLS: blue, green, red, narrow NIR, SWIR 1, SWIR 2), and their mean
#: and standard deviation in reflectance times 10,000 over its pretraining data.
ROLES = ("blue", "green", "red", "nir", "swir1", "swir2")
MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)
#: Band names of a cube (any case) that hold each of them; ``band_map`` adds others.
ALIASES = {
    "blue": ("blue", "b02", "b2"), "green": ("green", "b03", "b3"), "red": ("red", "b04", "b4"),
    "nir": ("nir08", "b8a", "nir", "b08", "b8"), "swir1": ("swir1", "swir16", "b11"),   # narrow NIR first, as HLS
    "swir2": ("swir2", "swir22", "b12"),
}
_HUB = "https://huggingface.co/ibm-nasa-geospatial"
#: The published models: repository, revision, file, SHA-256, size; width, depth, heads,
#: patch side; whether they encode the dates and the location (TL).
VARIANTS = {
    "tiny-TL": ("Prithvi-EO-2.0-tiny-TL", "335eadc2c45ad5abe7bd307223e1c48c5b60c41b", "Prithvi_EO_V2_tiny_TL.pt",
                "d47326db9bad502b611f73e3e3f3a0e68b7b82640d67c22c795417f7209f8d70", 192, 12, 3, 16, True),
    "100M-TL": ("Prithvi-EO-2.0-100M-TL", "2c84e383194986040f883cc43d7869002c425e1b", "Prithvi_EO_V2_100M_TL.pt",
                "d45406d5fc51af1d9657d48f2e2c3ff077408a2e1113f9a242889a4fe4469b17", 768, 12, 12, 16, True),
    "300M": ("Prithvi-EO-2.0-300M", "9eb1b1102806593963daa333bcc491b1c6f8562f", "Prithvi_EO_V2_300M.pt",
             "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7", 1024, 24, 16, 16, False),
    "300M-TL": ("Prithvi-EO-2.0-300M-TL", "63adbd39c271da4c42f447e69b1a7c91a338cdc9", "Prithvi_EO_V2_300M_TL.pt",
                "3629cedfbb350faafcb0dac902ae0d3c927e25ce8d9e0024aa1276ec66956ddb", 1024, 24, 16, 16, True),
    "600M": ("Prithvi-EO-2.0-600M", "2a5cf6cedea1c0b31734021e2a5ac90082825414", "Prithvi_EO_V2_600M.pt",
             "26a3f387ac6fc8e5eee8eb1986e6b9e99f36aa4d03aeb7fd5722feee5d5f88d0", 1280, 32, 16, 14, False),
    "600M-TL": ("Prithvi-EO-2.0-600M-TL", "3d72adc6dfc4cc3862bf4f41da8c83db267193c9", "Prithvi_EO_V2_600M_TL.pt",
                "7b92c53b0204a76bb775bd8930f045e05776251caa8c83f7367ed0b75b594702", 1280, 32, 16, 14, True),
}
_PRETRAINED_FRAMES, _PRETRAINED_SIZE = 4, 224


def pretrained_weights(variant: str):
    """The file of a variant's weights (``zeit.ai._weights.fetch``: downloaded once,
    ``ZEIT_WEIGHTS`` for a local copy)."""
    from ._weights import fetch

    repo, rev, name, sha = VARIANTS[variant][:4]
    return fetch(f"{_HUB}/{repo}/resolve/{rev}/{name}", sha, f"the weights of {repo}")


def match_bands(bands: Sequence[str], band_map: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Prithvi band -> index of the cube band that holds it; all six are needed."""
    band_map = dict(band_map or {})
    unknown = set(band_map) - set(ROLES)
    if unknown:
        raise ValueError(f"band_map keys are Prithvi's bands {list(ROLES)}, got {sorted(unknown)}")
    lower = {str(b).lower(): i for i, b in enumerate(bands)}
    found = {}
    for role in ROLES:
        if role in band_map:
            if band_map[role] not in list(bands):
                raise ValueError(f"band_map gives {band_map[role]!r} for {role}, but the bands are {list(bands)}")
            found[role] = list(bands).index(band_map[role])
            continue
        hit = next((lower[a] for a in ALIASES[role] if a in lower and lower[a] not in found.values()), None)
        if hit is not None:
            found[role] = hit
    missing = [r for r in ROLES if r not in found]
    if missing:
        raise ValueError(f"Prithvi takes the six HLS bands {list(ROLES)}; the cube has {list(bands)}, without "
                         f"{missing}. Name them so, or pass band_map={{band: cube band}}")
    return found


# ---------------------------------------------------------------------------
# The encoder, as the authors' (prithvi_mae.py, PrithviViT), with the names of its weights
# ---------------------------------------------------------------------------

def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=float) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1).astype(float), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _sincos_3d(dim: int, grid) -> np.ndarray:
    """Positions of a (t, h, w) grid of tokens, with a row of zeros first (the class token)."""
    t, h, w = grid
    w_e = np.tile(_sincos_1d(dim // 16 * 6, np.arange(w)), (t * h, 1))
    h_e = np.tile(np.repeat(_sincos_1d(dim // 16 * 6, np.arange(h)), w, axis=0), (t, 1))
    t_e = np.repeat(_sincos_1d(dim // 16 * 4, np.arange(t)), h * w, axis=0)
    return np.concatenate([np.zeros([1, dim]), np.concatenate((w_e, h_e, t_e), axis=1)], axis=0)


def _sincos_torch(dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=pos.dtype, device=pos.device)
    omega = 1.0 / 10000 ** (omega / (dim / 2.0))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class _PatchEmbed(nn.Module):
    def __init__(self, patch: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch, patch), stride=(1, patch, patch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class _CoordsEncoder(nn.Module):
    """Two coordinates (year and day of year; latitude and longitude) as sines and cosines,
    half the width each, times a learned scale."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        shape = coords.shape[:-1] + (-1,)
        a = _sincos_torch(half, coords[..., 0].flatten()).reshape(shape)
        b = _sincos_torch(self.embed_dim - half, coords[..., 1].flatten()).reshape(shape)
        return self.scale * torch.cat([a, b], dim=-1)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _Encoder(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, patch: int, coords: bool):
        super().__init__()
        self.embed_dim, self.patch = embed_dim, patch
        self.grid = (_PRETRAINED_FRAMES, _PRETRAINED_SIZE // patch, _PRETRAINED_SIZE // patch)
        self.patch_embed = _PatchEmbed(patch, len(ROLES), embed_dim)
        if coords:
            self.temporal_embed_enc = _CoordsEncoder(embed_dim)
            self.location_embed_enc = _CoordsEncoder(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.from_numpy(_sincos_3d(embed_dim, self.grid)).float()[None])
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _positions(self, t: int, h: int, w: int) -> torch.Tensor:
        """The positions of another grid of tokens, as the authors make them: computed again
        for another number of dates, interpolated (bicubic) for another size."""
        if (t, h, w) == self.grid:
            return self.pos_embed
        pos = self.pos_embed
        if t != self.grid[0]:
            pos = torch.from_numpy(_sincos_3d(self.embed_dim, (t,) + self.grid[1:])).float()[None].to(pos.device)
        if (h, w) == self.grid[1:]:
            return pos
        cls, grid = pos[:, :1], pos[:, 1:].reshape(t, *self.grid[1:], self.embed_dim).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=True)
        return torch.cat((cls, grid.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)), dim=1)

    def forward(self, x: torch.Tensor, temporal: Optional[torch.Tensor] = None,
                location: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, 6, dates, H, W), standardized; temporal: (batch, dates, 2) year and day
        of year; location: (batch, 2) latitude and longitude. The tokens of the last layer,
        the class token first: (batch, 1 + dates * H/p * W/p, width)."""
        t, h, w = x.shape[2], x.shape[3] // self.patch, x.shape[4] // self.patch
        pos = self._positions(t, h, w)
        x = self.patch_embed(x) + pos[:, 1:]
        if temporal is not None and hasattr(self, "temporal_embed_enc"):   # each date's, on its tokens
            x = x + torch.repeat_interleave(self.temporal_embed_enc(temporal), h * w, dim=1)
        if location is not None and hasattr(self, "location_embed_enc"):
            x = x + self.location_embed_enc(location[:, None])
        x = torch.cat(((self.cls_token + pos[:, :1]).expand(x.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# On the bands of a cube, with a decoder to a class per pixel
# ---------------------------------------------------------------------------

class _Decoder(nn.Module):
    """The tokens of all dates side by side (time folded into the channels, as TerraTorch
    does), doubled in size back to the pixels; then, joined to the bands of the pixels
    themselves, to a class per pixel. A token is 16 x 16 pixels: from the tokens alone, an
    edge inside one cannot be placed to the pixel."""

    def __init__(self, in_channels: int, patch: int, num_classes: int, pixel_channels: int, width: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        ups, c = [], width
        for _ in range(math.ceil(math.log2(patch))):
            out = max(c // 2, 32)
            ups.append(nn.Sequential(nn.ConvTranspose2d(c, out, 2, stride=2), nn.BatchNorm2d(out), nn.ReLU(inplace=True)))
            c = out
        self.up = nn.Sequential(*ups)
        self.pixel = nn.Sequential(nn.Conv2d(pixel_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(c + 32, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d(c, num_classes, 1)

    def forward(self, tokens: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        x = self.up(self.proj(tokens))
        if x.shape[-2:] != image.shape[-2:]:   # a patch of 14: 16 times, then back to the pixels
            x = F.interpolate(x, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.fuse(torch.cat([x, self.pixel(image)], 1)))


class Prithvi(nn.Module):
    """Prithvi-EO-2.0's encoder and a decoder to per-pixel classes, on windows of a cube
    ``(batch, bands, dates, H, W)`` in its own units, NaN where missing; ``H`` and ``W``
    multiples of the patch (16; 14 for the 600M models).

    Parameters
    ----------
    bands
        The cube's bands, in order; the six HLS bands are found by name (``ALIASES``).
    n_dates
        Dates of the windows (Prithvi was pretrained on 4; any number works).
    num_classes
        Classes of the decoder.
    variant
        ``"tiny-TL"``, ``"100M-TL"`` (default), ``"300M"``, ``"300M-TL"``, ``"600M"`` or
        ``"600M-TL"``: the published models (TL: they also encode the dates and the place).
    dates
        The dates (anything ``pandas.Timestamp`` reads), for the TL models.
    latlon
        Latitude and longitude (degrees) of the place, for the TL models.
    band_map
        ``{Prithvi band: cube band}`` for bands whose names are not recognized.
    reflectance_scale
        Factor bringing the bands to reflectance times 10,000 (``1e4`` for reflectance
        from 0 to 1).
    freeze_encoder
        Train the decoder only (the encoder's weights stay as pretrained).
    pretrained
        Start from the authors' weights (``pretrained_weights``: 129 MB for tiny-TL, 455 MB
        for 100M-TL, 1.3 GB for 300M, 2.6 GB for 600M, downloaded once).
    """

    def __init__(self, bands: Sequence[str], n_dates: int, num_classes: int = 2, variant: str = "100M-TL",
                 dates: Optional[Sequence] = None, latlon: Optional[Sequence[float]] = None,
                 band_map: Optional[Dict[str, str]] = None, reflectance_scale: float = 1.0,
                 freeze_encoder: bool = False, pretrained: bool = True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}, got {variant!r}")
        if dates is not None and len(dates) != n_dates:
            raise ValueError(f"{len(dates)} dates for n_dates={n_dates}")
        width, depth, heads, patch, coords = VARIANTS[variant][4:]
        self.variant, self.patch, self.n_bands, self.n_dates = variant, patch, len(bands), int(n_dates)
        found = match_bands(bands, band_map)
        self.register_buffer("_src", torch.tensor([found[r] for r in ROLES]), persistent=False)
        self.register_buffer("_mean", torch.tensor(MEAN).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(STD).view(1, -1, 1, 1, 1), persistent=False)
        self.reflectance_scale = float(reflectance_scale)
        temporal = location = None
        if coords and dates is not None:
            import pandas as pd

            stamps = [pd.Timestamp(d) for d in dates]
            temporal = torch.tensor([[s.year, s.dayofyear] for s in stamps], dtype=torch.float32)
        if coords and latlon is not None:
            location = torch.tensor(list(latlon), dtype=torch.float32)
        self.register_buffer("_temporal", temporal, persistent=False)
        self.register_buffer("_location", location, persistent=False)
        self.encoder = _Encoder(width, depth, heads, patch, coords)
        if pretrained:
            state = torch.load(pretrained_weights(variant), map_location="cpu", weights_only=True)
            # the positions are computed, as the authors' inference does (not read from the file)
            state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
            state["pos_embed"] = self.encoder.pos_embed
            self.encoder.load_state_dict(state)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.head = _Decoder(width * self.n_dates, patch, num_classes, len(ROLES) * self.n_dates)

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """The six bands of ``(batch, bands, dates, H, W)``, standardized as Prithvi was
        trained; what is missing at Prithvi's mean (0)."""
        if x.shape[1] != self.n_bands or x.shape[2] != self.n_dates:
            raise ValueError(f"this Prithvi takes {self.n_dates} dates of {self.n_bands} bands, "
                             f"got {x.shape[2]} of {x.shape[1]}")
        if x.shape[3] % self.patch or x.shape[4] % self.patch:
            raise ValueError(f"Prithvi {self.variant} takes windows of a multiple of {self.patch} pixels, "
                             f"got {tuple(x.shape[3:])}")
        x = (x[:, self._src].float() * self.reflectance_scale - self._mean) / self._std
        return torch.nan_to_num(x, nan=0.0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        temporal = None if self._temporal is None else self._temporal.expand(b, -1, -1)
        location = None if self._location is None else self._location.expand(b, -1)
        return self.encoder(x, temporal, location)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The encoder's tokens of the last layer, the class token first."""
        return self._encode(self.standardize(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.standardize(x)
        b, _, t, height, width = x.shape
        tokens = self._encode(x)[:, 1:]
        h, w = height // self.patch, width // self.patch
        grid = tokens.reshape(b, t, h, w, -1).permute(0, 1, 4, 2, 3).reshape(b, -1, h, w)   # (t e) h w
        return self.head(grid, x.reshape(b, -1, height, width))


class GeoFoundationViT(Prithvi):
    """The name of ``Prithvi`` before zeit 1.7 (then a wrapper of Prithvi-100M through
    ``transformers``): ``model_id`` names a Prithvi-EO-2.0 repository."""

    def __init__(self, *args, model_id: Optional[str] = None, **kwargs):
        warnings.warn("zeit.ai.GeoFoundationViT is now zeit.ai.Prithvi (Prithvi-EO-2.0, without transformers)",
                      DeprecationWarning, stacklevel=2)
        if model_id is not None:
            name = str(model_id).rsplit("/", 1)[-1]
            variant = next((v for v, spec in VARIANTS.items() if spec[0] == name), None)
            if variant is None:
                raise ValueError(f"{model_id!r} is not a Prithvi-EO-2.0 model; zeit.ai.Prithvi takes "
                                 f"variant= one of {list(VARIANTS)}")
            kwargs["variant"] = variant
        super().__init__(*args, **kwargs)
