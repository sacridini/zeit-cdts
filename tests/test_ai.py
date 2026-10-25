import pytest

torch = pytest.importorskip("torch")
from zeit.ai.tempcnn import TempCNN
from zeit.ai.utae import LTAE, LightTAE, UTAE

def test_tempcnn():
    # Batch size 4, 6 bands, 24 timesteps
    x = torch.randn(4, 6, 24)
    model = TempCNN(in_channels=6, n_times=24, num_classes=5)

    out = model(x)
    assert out.shape == (4, 5) # 4 pixels, 5 classes


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
