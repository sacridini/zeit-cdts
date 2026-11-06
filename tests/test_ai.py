import math

import pytest

torch = pytest.importorskip("torch")
from zeit.ai.tempcnn import TempCNN
from zeit.ai.ltae import LTAE, LightTAE, positional_table
from zeit.ai.utae import UTAE

def test_tempcnn():
    # Batch size 4, 6 bands, 24 timesteps
    x = torch.randn(4, 6, 24)
    model = TempCNN(in_channels=6, n_times=24, num_classes=5)

    out = model(x)
    assert out.shape == (4, 5) # 4 pixels, 5 classes


def test_tempcnn_parameters_and_init():
    C, L, n = 3, 10, 4
    model = TempCNN(in_channels=C, n_times=L, num_classes=n)
    conv1, conv23 = 5 * C * 64 + 64, 5 * 64 * 64 + 64
    expected = conv1 + 2 * conv23 + 3 * 128 + (64 * L * 256 + 256) + 2 * 256 + (256 * n + n)
    assert sum(p.numel() for p in model.parameters()) == expected == 208132
    # the batch normalization's moving averages: 128 per layer of 64, 512 for the dense one, not trainable
    assert sum(b.numel() for name, b in model.named_buffers() if "running" in name) == 3 * 128 + 512
    # He normal on the hidden kernels (std sqrt(2 / fan_in)), Glorot uniform on the output (limit sqrt(6 / (in + out)))
    torch.manual_seed(0)
    big = TempCNN(in_channels=8, n_times=40, num_classes=6)
    w = big.conv[4].weight                          # the second convolution: fan_in 64 * 5
    assert abs(w.std().item() / (2 / (64 * 5)) ** 0.5 - 1) < 0.05
    w = big.dense[1].weight                         # fan_in 64 * 40
    assert abs(w.std().item() / (2 / (64 * 40)) ** 0.5 - 1) < 0.05
    limit = (6 / (256 + 6)) ** 0.5
    assert big.out.weight.abs().max().item() <= limit and big.out.weight.abs().max().item() > 0.9 * limit
    assert all((m.bias == 0).all() for m in big.modules() if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)))
    # same padding, logits, the L2 penalty of the authors, and a deterministic evaluation
    assert model.conv(torch.randn(2, C, L)).shape == (2, 64, L)
    penalty = model.l2_penalty()
    assert torch.isclose(penalty, 1e-6 * sum((k ** 2).sum() for k in model.kernels()))
    assert len(list(model.kernels())) == 5
    model.eval()
    x = torch.randn(3, C, L)
    assert torch.equal(model(x), model(x))


def test_positional_table_follows_the_formula():
    days = [0, 1, 16, 400]
    table = positional_table(days, d=16, T=1000, repeat=4)
    assert table.shape == (4, 64)
    assert torch.allclose(table[0, :16], torch.tensor([0.0, 1.0] * 8))      # day 0: sin 0, cos 0, ...
    for i, p in enumerate(days):
        for j in range(16):
            ang = p / 1000 ** (2 * (j // 2) / 16)
            want = math.sin(ang) if j % 2 == 0 else math.cos(ang)
            assert abs(table[i, j].item() - want) < 1e-5
    assert torch.equal(table[:, :16], table[:, 16:32]) and torch.equal(table[:, :16], table[:, 48:])


def test_ltae_attention_and_parameters():
    days = [0, 10, 25, 40, 70]
    model = LTAE(in_channels=8, day_offsets=days, n_head=4, d_k=4, n_neurons=(32, 16), d_model=32)
    # in_norm 16, inconv 288, layer norm over (32, 5) 320, Q 16, keys 528, MLP 528 + batch norm 32, out norm 32
    assert sum(p.numel() for p in model.parameters()) == 16 + 288 + 320 + 16 + 528 + 560 + 32 == 1760
    assert model.positions.shape == (5, 32)
    model.eval()
    x = torch.randn(6, 5, 8)
    out, attn = model(x, return_attention=True)
    assert out.shape == (6, 16) and attn.shape == (6, 4, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(6, 4), atol=1e-6) and (attn >= 0).all()
    assert torch.equal(out, model(x))
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 8))                 # another number of dates
    # without a projection the width is in_channels, and the heads' values are the channels' groups
    plain = LTAE(in_channels=16, day_offsets=days, n_head=4, d_k=4, n_neurons=(16, 8), d_model=None, init="default")
    plain.eval()
    z = torch.randn(2, 5, 16)
    o, a = plain.attention(z)
    values = z.view(2, 5, 4, 4)
    assert torch.allclose(o.view(2, 4, 4), torch.einsum("bht,bthc->bhc", a, values), atol=1e-6)


def test_ltae_authors_defaults_and_init():
    model = LTAE(in_channels=128, day_offsets=list(range(0, 240, 10)))
    assert (model.n_head, model.d_model, model.attention.d_k) == (16, 256, 8) and model.positions.shape == (24, 256)
    assert model.out_norm.normalized_shape == (128,) and model.dropout.p == 0.2 and model.attention.dropout.p == 0.1
    assert abs(model.attention.Q.std().item() / (2 / 8) ** 0.5 - 1) < 0.3
    fc = model.attention.fc1_k.weight             # Xavier normal: std sqrt(2 / (256 + 128))
    assert abs(fc.std().item() / (2 / (256 + 128)) ** 0.5 - 1) < 0.05
    with pytest.raises(ValueError):
        LTAE(in_channels=128, day_offsets=[0, 1], n_head=5)
    with pytest.raises(ValueError):
        LTAE(in_channels=128, day_offsets=[0, 1], n_neurons=(100, 8))


def test_ltae_master_query_is_a_weighted_mean():
    model = LTAE(in_channels=8, day_offsets=[0, 5, 9], n_head=2, d_k=2, n_neurons=(8, 4), d_model=None, init="default")
    model.eval()
    with torch.no_grad():
        model.attention.Q.zero_()                 # equal scores: a plain mean over the dates
    z = torch.randn(3, 3, 8)
    out, attn = model.attention(z)
    assert torch.allclose(attn, torch.full((3, 2, 3), 1 / 3), atol=1e-6)
    assert torch.allclose(out, z.mean(1), atol=1e-6)


def test_ltae():
    day_offsets = list(range(0, 24 * 16, 16))  # 24 steps, 16-day composites
    model = LTAE(in_channels=128, day_offsets=day_offsets)

    x = torch.randn(4, 24, 128)  # (batch, seq_len, in_channels)
    out = model(x)
    assert out.shape == (4, 128)  # n_neurons[-1] default


def test_lighttae():
    day_offsets = list(range(0, 24 * 16, 16))
    model = LightTAE(n_bands=6, day_offsets=day_offsets, n_labels=5)

    x = torch.randn(4, 24, 6)  # (batch, n_times, n_bands)
    out = model(x)
    assert out.shape == (4, 5)


def test_utae():
    model = UTAE(
        input_dim=6,
        encoder_widths=[8, 16, 32],
        decoder_widths=[8, 16, 32],
        out_conv=[16, 3],
        n_head=4,
        d_model=32,
        d_k=4,
    )
    model.eval()

    x = torch.randn(2, 5, 6, 16, 16)  # (batch, seq_len, input_dim, H, W)
    batch_positions = torch.arange(5, dtype=torch.float32).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        out = model(x, batch_positions=batch_positions)
    assert out.shape == (2, 3, 16, 16)


def test_prithvi_keeps_the_input_size():
    """A class per pixel of the window, from tokens of 16 x 16 (14 x 14 for the 600M models,
    brought back by interpolation)."""
    from zeit.ai.prithvi import Prithvi, _Decoder

    model = Prithvi(["blue", "green", "red", "nir", "swir1", "swir2"], n_dates=3, num_classes=4,
                    variant="tiny-TL", pretrained=False)
    assert model(torch.rand(2, 6, 3, 48, 32)).shape == (2, 4, 48, 32)
    assert _Decoder(16, 14, 3, 12)(torch.rand(1, 16, 4, 5), torch.rand(1, 12, 56, 70)).shape == (1, 3, 56, 70)
