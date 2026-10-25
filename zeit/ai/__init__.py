try:
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError("zeit.ai needs PyTorch, which zeit installs only when asked since 2.3: pip install "
                      "\"zeit-cdts[ai]\" (for a GPU, install torch first as pytorch.org says for its CUDA "
                      "version)") from e

from .siamese import SiameseChangeDetector
from .utae import UTAE, LTAE, LightTAE
from .tempcnn import TempCNN
from .presto import Presto, presto_embeddings
from .prithvi import Prithvi, GeoFoundationViT
from .dataset import STACCubeDataset
from .losses import FocalLoss, TverskyLoss, ContrastiveSiameseLoss
from .._som import SOM
from .pipeline import SampleSet, samples, train, predict, save, load

__all__ = ['SiameseChangeDetector', 'UTAE', 'LTAE', 'LightTAE', 'TempCNN', 'Presto', 'presto_embeddings', 'Prithvi', 'GeoFoundationViT', 'STACCubeDataset', 'FocalLoss', 'TverskyLoss', 'ContrastiveSiameseLoss', 'SOM',
           'SampleSet', 'samples', 'train', 'predict', 'save', 'load']