import math
import torch
import torch.nn as nn


class _PositionalEncoderUTAE(nn.Module):
    """Sinusoidal positional encoding computed dynamically from per-sample
    date values (unlike `zeit.ai.ltae.positional_table`, which bakes a fixed
    timeline into a buffer at construction time - U-TAE's encoder supports a
    different, possibly irregular, set of dates per forward call). Matches
    the official U-TAE repo's `PositionalEncoder`
    (VSainteuf/utae-paps, src/backbones/positional_encoding.py)."""

    def __init__(self, d: int, T: int = 1000, repeat: int = None, offset: int = 0) -> None:
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.register_buffer("denom", denom)

    def forward(self, batch_positions: "torch.Tensor") -> "torch.Tensor":
        # batch_positions: (batch, seq_len) -> (batch, seq_len, d [* repeat])
        sinusoid_table = batch_positions[:, :, None] / self.denom[None, None, :]
        sinusoid_table = sinusoid_table.clone()
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])
        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table] * self.repeat, dim=-1)
        return sinusoid_table


class _ScaledDotProductAttentionUTAE(nn.Module):
    """Matches the official U-TAE repo's `ScaledDotProductAttention`
    (src/backbones/ltae.py), which additionally supports masking out padded
    timesteps (`pad_mask`), which the pixel-series L-TAE (`zeit.ai.ltae`, fixed
    dates) does not need."""

    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, keys, values, pad_mask=None):
        attn = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        return torch.matmul(attn, values), attn


class _MultiHeadAttentionUTAE(nn.Module):
    """Learned master-query multi-head attention with padding support,
    matching the official U-TAE repo's `MultiHeadAttention`
    (src/backbones/ltae.py). Same "master query" trick as
    the one in `zeit.ai.ltae` (also the authors' code), kept separate to stay a
    faithful, independently-verifiable port of its own reference."""

    def __init__(self, n_head: int, d_k: int, d_in: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.Q, mean=0.0, std=math.sqrt(2.0 / d_k))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0.0, std=math.sqrt(2.0 / d_k))

        self.attention = _ScaledDotProductAttentionUTAE(temperature=math.sqrt(d_k))

    def forward(self, v: "torch.Tensor", pad_mask: "torch.Tensor" = None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.shape

        q = self.Q.unsqueeze(1).expand(n_head, sz_b, d_k).reshape(n_head * sz_b, d_k)

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(n_head * sz_b, seq_len, d_k)

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(n_head, 1)

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)

        output, attn = self.attention(q, k, v, pad_mask=pad_mask)
        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(2)
        output = output.view(n_head, sz_b, 1, d_in // n_head).squeeze(2)
        return output, attn


class _LTAE2d(nn.Module):
    """Lightweight Temporal Attention Encoder for image time series: applies
    a shared per-pixel L-TAE across every spatial position of a sequence of
    2D feature maps, fusing them into a single feature map (+ optional
    per-head/per-timestep attention maps used to weight U-TAE's decoder skip
    connections). Ported layer-for-layer from the official U-TAE repo's
    `LTAE2d` (VSainteuf/utae-paps, src/backbones/ltae.py; Garnot &
    Landrieu, 2021, arXiv:2007.00586 for the underlying L-TAE mechanism)."""

    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 16,
        d_k: int = 4,
        mlp: "list[int]" = (256, 128),
        dropout: float = 0.2,
        d_model: int = 256,
        T: int = 1000,
        return_att: bool = False,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        mlp = list(mlp)
        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = _PositionalEncoderUTAE(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = _MultiHeadAttentionUTAE(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=mlp[-1])

        mlp_layers = []
        for i in range(len(mlp) - 1):
            mlp_layers += [nn.Linear(mlp[i], mlp[i + 1]), nn.BatchNorm1d(mlp[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*mlp_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor", batch_positions: "torch.Tensor" = None, pad_mask: "torch.Tensor" = None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).repeat(1, 1, h).unsqueeze(-1).repeat(1, 1, 1, w)
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = batch_positions.unsqueeze(-1).repeat(1, 1, h).unsqueeze(-1).repeat(1, 1, 1, w)
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)  # concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)  # head x b x t x h x w

        if self.return_att:
            return out, attn
        return out


class _TemporallySharedBlock(nn.Module):
    """Adds `smart_forward()`: folds a (B,T,C,H,W) tensor into (B*T,C,H,W)
    and applies the (time-shared) conv block to every frame, optionally
    skipping frames that are entirely padding (`pad_value`) to save compute.
    Matches the official U-TAE repo's `TemporallySharedBlock`."""

    def __init__(self, pad_value: float = None) -> None:
        super().__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if len(x.shape) == 4:
            return self.forward(x)

        b, t, c, h, w = x.shape
        if self.pad_value is not None:
            dummy = torch.zeros(x.shape, device=x.device).float()
            self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

        out = x.view(b * t, c, h, w)
        if self.pad_value is not None:
            pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            if pad_mask.any():
                temp = torch.ones(self.out_shape, device=x.device, requires_grad=False) * self.pad_value
                temp[~pad_mask] = self.forward(out[~pad_mask])
                out = temp
            else:
                out = self.forward(out)
        else:
            out = self.forward(out)
        _, c, h, w = out.shape
        return out.view(b, t, c, h, w)


class _ConvLayer(nn.Module):
    def __init__(self, nkernels, norm="batch", k=3, s=1, p=1, n_groups=4, last_relu=True, padding_mode="reflect"):
        super().__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(nn.Conv2d(
                nkernels[i], nkernels[i + 1], kernel_size=k, padding=p, stride=s, padding_mode=padding_mode,
            ))
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))
            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class _ConvBlock(_TemporallySharedBlock):
    def __init__(self, nkernels, pad_value=None, norm="batch", last_relu=True, padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.conv = _ConvLayer(nkernels=nkernels, norm=norm, last_relu=last_relu, padding_mode=padding_mode)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.conv(x)


class _DownConvBlock(_TemporallySharedBlock):
    """Strided-conv downsampling stage + a residual conv pair, shared across
    time (see `_TemporallySharedBlock`)."""

    def __init__(self, d_in, d_out, k, s, p, pad_value=None, norm="batch", padding_mode="reflect"):
        super().__init__(pad_value=pad_value)
        self.down = _ConvLayer(nkernels=[d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode)
        self.conv1 = _ConvLayer(nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = _ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out = self.down(x)
        out = self.conv1(out)
        return out + self.conv2(out)


class _UpConvBlock(nn.Module):
    """Transposed-conv upsampling stage, concatenated with a 1x1-projected
    skip connection, + a residual conv pair."""

    def __init__(self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(nn.Conv2d(d, d, kernel_size=1), nn.BatchNorm2d(d), nn.ReLU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = _ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode)
        self.conv2 = _ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode)

    def forward(self, x: "torch.Tensor", skip: "torch.Tensor") -> "torch.Tensor":
        out = self.up(x)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        return out + self.conv2(out)


class _TemporalAggregator(nn.Module):
    """Aggregates a stack of per-timestep skip-connection feature maps into
    one, weighted by the L-TAE attention masks (resampled to the skip's
    spatial resolution) - the mechanism that propagates the bottleneck's
    learned temporal attention to every decoder scale, not just the
    bottleneck itself. Matches the official U-TAE repo's
    `Temporal_Aggregator`."""

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: "torch.Tensor", pad_mask: "torch.Tensor" = None, attn_mask: "torch.Tensor" = None):
        has_padding = pad_mask is not None and pad_mask.any()

        if self.mode == "att_group":
            n_heads, b, t, h, w = attn_mask.shape
            attn = attn_mask.view(n_heads * b, t, h, w)
            if x.shape[-2] > w:
                attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
            else:
                attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
            attn = attn.view(n_heads, b, t, *x.shape[-2:])
            if has_padding:
                attn = attn * (~pad_mask).float()[None, :, :, None, None]
            out = torch.stack(x.chunk(n_heads, dim=2))  # h x B x T x C/h x H x W
            out = attn[:, :, :, None, :, :] * out
            out = out.sum(dim=2)  # sum over time -> h x B x C/h x H x W
            return torch.cat([group for group in out], dim=1)  # -> B x C x H x W
        elif self.mode == "att_mean":
            attn = attn_mask.mean(dim=0)  # average over heads -> B x T x H x W
            attn = nn.Upsample(size=x.shape[-2:], mode="bilinear", align_corners=False)(attn)
            if has_padding:
                attn = attn * (~pad_mask).float()[:, :, None, None]
            return (x * attn[:, :, None, :, :]).sum(dim=1)
        elif self.mode == "mean":
            if has_padding:
                out = x * (~pad_mask).float()[:, :, None, None, None]
                return out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
            return x.mean(dim=1)
        raise ValueError(f"Unknown aggregation mode: {self.mode!r}")


class UTAE(nn.Module):
    """
    U-Net with Temporal Attention Encoder for spatio-temporal segmentation of
    satellite image time series, ported layer-for-layer from the official
    reference implementation:

    Garnot, V.S.F. & Landrieu, L. (2021). "Panoptic Segmentation of
    Satellite Image Time Series with Convolutional Temporal Attention
    Networks". ICCV 2021. doi:10.1109/ICCV48922.2021.00483.
    Code: https://github.com/VSainteuf/utae-paps (MIT License).

    A multi-scale convolutional U-Net encodes each frame independently
    (weights shared across time - see `_TemporallySharedBlock`), an L-TAE
    (`_LTAE2d`) fuses the bottleneck features across time into a single
    feature map plus per-head/per-timestep attention maps, and those same
    attention maps (resampled per scale, see `_TemporalAggregator`) weight
    the temporal aggregation of every decoder skip connection - propagating
    the learned "when does this pixel matter" signal to every resolution,
    not just the bottleneck.

    Verified against the official implementation: building both with
    identical weights (via `load_state_dict`, state_dict keys match by
    name with no translation table) and the same input reproduces its
    output bit-for-bit exactly - including the padded-sequence code path
    (`pad_value`/`pad_mask`). See the [UTAE tutorial](../../docs/tutorials/utae.md)
    for the validation methodology (this cross-check isn't part of the
    pytest suite, since it needs the reference repo cloned locally, not a
    pip dependency).
    """

    def __init__(
        self,
        input_dim: int,
        encoder_widths: "list[int]" = (64, 64, 64, 128),
        decoder_widths: "list[int]" = (32, 32, 64, 128),
        out_conv: "list[int]" = (32, 20),
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        encoder: bool = False,
        return_maps: bool = False,
        pad_value: float = 0,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        encoder_widths = list(encoder_widths)
        decoder_widths = list(decoder_widths) if decoder_widths is not None else None
        out_conv = list(out_conv)

        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        self.stack_dim = sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = _ConvBlock(
            nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value, norm=encoder_norm, padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList([
            _DownConvBlock(
                d_in=encoder_widths[i], d_out=encoder_widths[i + 1],
                k=str_conv_k, s=str_conv_s, p=str_conv_p,
                pad_value=pad_value, norm=encoder_norm, padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        ])
        self.up_blocks = nn.ModuleList([
            _UpConvBlock(
                d_in=decoder_widths[i], d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k, s=str_conv_s, p=str_conv_p,
                norm="batch", padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        ])
        self.temporal_encoder = _LTAE2d(
            in_channels=encoder_widths[-1], d_model=d_model, n_head=n_head,
            mlp=[d_model, encoder_widths[-1]], return_att=True, d_k=d_k,
        )
        self.temporal_aggregator = _TemporalAggregator(mode=agg_mode)
        self.out_conv = _ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)

    def forward(self, input: "torch.Tensor", batch_positions: "torch.Tensor" = None, return_att: bool = False):
        # input: (batch, seq_len, input_dim, H, W); batch_positions: (batch, seq_len)
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # (batch, seq_len)

        out = self.in_conv.smart_forward(input)
        feature_maps = [out]
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        out, att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att)
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        out = self.out_conv(out)
        if return_att:
            return out, att
        if self.return_maps:
            return out, maps
        return out
