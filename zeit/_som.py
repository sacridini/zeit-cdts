import warnings

import numpy as np
from . import _core

_LR_DECAYS = {'asymptotic_decay': 0, 'inverse_decay_to_zero': 1, 'linear_decay_to_zero': 2}
_SIGMA_DECAYS = {'asymptotic_decay': 0, 'inverse_decay_to_one': 1, 'linear_decay_to_one': 2}
_NEIGHBORHOODS = {'gaussian': 0, 'mexican_hat': 1, 'bubble': 2, 'triangle': 3}
_ALGORITHMS = ('online', 'batch')


class SOM:
    """
    Self-Organizing Map (SOM) accelerated by C++ and OpenMP.

    The implementation reproduces Python ``minisom`` (MiniSom 2.3.x): with the
    same ``random_seed`` and arguments, the initial weights, sample ordering,
    neighborhood/decay functions and weight updates are identical, so the
    trained codebook matches ``MiniSom`` to machine precision.

    Two training algorithms are available:

    - ``algorithm='online'`` (default): the classic sample-by-sample Kohonen
      update, equivalent to ``MiniSom.train``. ``num_iters`` is the number of
      single-sample updates (or epochs with ``use_epochs=True``).
    - ``algorithm='batch'``: Batch SOM (Kohonen, 2013), equivalent to
      ``MiniSom.train_batch_offline``. ``num_iters`` is the number of passes
      over the whole dataset; the best-matching-unit search and the weight
      accumulation run in parallel with OpenMP (``n_jobs``).
    """

    Y_HEX_CONV_FACTOR = (3.0 / 2.0) / np.sqrt(3)

    def __init__(self, x: int, y: int, input_len: int, sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 decay_function: str = 'asymptotic_decay',
                 neighborhood_function: str = 'gaussian',
                 topology: str = 'rectangular',
                 random_seed: int = 42,
                 sigma_decay_function: str = 'asymptotic_decay'):
        if decay_function not in _LR_DECAYS:
            raise ValueError(f"{decay_function} not supported. Functions available: {', '.join(_LR_DECAYS)}")
        if sigma_decay_function not in _SIGMA_DECAYS:
            raise ValueError(f"{sigma_decay_function} not supported. Functions available: {', '.join(_SIGMA_DECAYS)}")
        if neighborhood_function not in _NEIGHBORHOODS:
            raise ValueError(f"{neighborhood_function} not supported. Functions available: {', '.join(_NEIGHBORHOODS)}")
        if topology not in ('rectangular', 'hexagonal'):
            raise ValueError(f"{topology} not supported only hexagonal and rectangular available")
        if sigma > np.sqrt(x * x + y * y):
            warnings.warn('Warning: sigma might be too high for the dimension of the map.')
        if neighborhood_function in ('triangle', 'bubble') and (divmod(sigma, 1)[1] != 0 or sigma < 1):
            warnings.warn('sigma should be an integer >=1 when triangle or bubble are used as neighborhood function')

        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay_function = decay_function
        self.sigma_decay_function = sigma_decay_function
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.random_seed = random_seed
        self._random_generator = np.random.RandomState(random_seed)

        # Same draws as MiniSom.__init__: uniform [-1, 1), unit-normalized.
        weights = self._random_generator.rand(x, y, input_len) * 2 - 1
        weights /= np.linalg.norm(weights, axis=-1, keepdims=True)
        self.weights = np.ascontiguousarray(weights, dtype=np.float64)

        xx, yy = np.meshgrid(np.arange(x), np.arange(y))
        xx = xx.astype(float)
        yy = yy.astype(float)
        if topology == 'hexagonal':
            xx[::-2] -= 0.5
            yy *= self.Y_HEX_CONV_FACTOR
        # Neuron (i, j) euclidean coordinates, flattened in (i, j) order.
        self._xx = np.ascontiguousarray(xx.T).ravel()
        self._yy = np.ascontiguousarray(yy.T).ravel()
        self._is_trained = False

    def _check_data(self, data: np.ndarray) -> np.ndarray:
        data = np.ascontiguousarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Data must be a 2D array [Samples, Features]")
        if data.shape[1] != self.input_len:
            raise ValueError(f"Received {data.shape[1]} features, expected {self.input_len}.")
        if len(data) == 0:
            raise ValueError("Data must contain at least one sample")
        if not np.isfinite(data).all():
            raise ValueError("Data contains NaN or infinite values")
        return data

    def get_weights(self) -> np.ndarray:
        """Codebook of shape (x, y, input_len)."""
        return self.weights

    def random_weights_init(self, data: np.ndarray):
        """Initializes the weights by picking random samples from data (MiniSom.random_weights_init)."""
        data = self._check_data(data)
        # One vectorized draw yields the same stream as MiniSom's per-neuron randint calls.
        picks = self._random_generator.randint(len(data), size=self.x * self.y)
        self.weights = np.ascontiguousarray(data[picks].reshape(self.x, self.y, self.input_len))

    def pca_weights_init(self, data: np.ndarray):
        """Initializes the weights to span the first two principal components (MiniSom.pca_weights_init)."""
        if self.input_len == 1:
            raise ValueError('The data needs at least 2 features for pca initialization')
        data = self._check_data(data)
        if self.x == 1 or self.y == 1:
            warnings.warn('PCA initialization inappropriate: One of the dimensions of the map is 1.')
        eigvals, eigvecs = np.linalg.eigh(np.cov(data, rowvar=False))
        order = np.argsort(eigvals)[::-1]
        pc1 = eigvecs[:, order[0]]
        pc2 = eigvecs[:, order[1]]
        data_mean = np.mean(data, axis=0)
        for i, c1 in enumerate(np.linspace(-1, 1, self.x)):
            for j, c2 in enumerate(np.linspace(-1, 1, self.y)):
                self.weights[i, j] = data_mean + c1 * pc1 + c2 * pc2

    def train(self, data: np.ndarray, num_iters: int, n_jobs: int = -1,
              algorithm: str = 'online', random_order: bool = False,
              use_epochs: bool = False):
        """
        Trains the SOM, continuing from the current weights.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (n_samples, input_len).
        num_iters : int
            ``online``: number of single-sample updates (epochs when
            ``use_epochs=True``), as in ``MiniSom.train``.
            ``batch``: number of passes over the whole dataset, as in
            ``MiniSom.train_batch_offline``.
        n_jobs : int
            OpenMP threads for the batch algorithm (-1 = all cores but one).
            The online update is sequential by definition and ignores it.
        algorithm : {'online', 'batch'}
        random_order : bool
            ``online`` only: pick samples in random order (``MiniSom.train_random``).
        use_epochs : bool
            ``online`` only: ``num_iters`` counts epochs over the data.
        """
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}")
        if num_iters < 1:
            raise ValueError('num_iteration must be > 1')
        data = self._check_data(data)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        args = (self.learning_rate, self.sigma, _LR_DECAYS[self.decay_function],
                _SIGMA_DECAYS[self.sigma_decay_function],
                _NEIGHBORHOODS[self.neighborhood_function], self._xx, self._yy)

        if algorithm == 'batch':
            _core.som.train_batch(self.weights, data, int(num_iters), *args, n_jobs=n_jobs)
        else:
            # Sample order exactly as MiniSom's _build_iteration_indexes.
            if use_epochs:
                order = np.arange(len(data))
                if random_order:
                    self._random_generator.shuffle(order)
            else:
                order = np.arange(num_iters) % len(data)
                if random_order:
                    self._random_generator.shuffle(order)
            _core.som.train_online(self.weights, data, order.astype(np.int64), int(num_iters),
                                   bool(use_epochs), *args)
        self._is_trained = True

    def predict(self, data: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """Flat index (i * y + j) of the best matching unit of each sample (MiniSom.winner)."""
        if not self._is_trained:
            raise ValueError("SOM is not trained yet.")
        data = self._check_data(data)
        return _core.som.predict_bmus(data, self.weights, n_jobs)

    def winner(self, x: np.ndarray):
        """Grid coordinates (i, j) of the best matching unit of a single sample."""
        idx = int(_core.som.predict_bmus(self._check_data(np.atleast_2d(x)), self.weights, 1)[0])
        return np.unravel_index(idx, (self.x, self.y))

    def quantization(self, data: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """Codebook vector of the best matching unit of each sample."""
        data = self._check_data(data)
        bmus = _core.som.predict_bmus(data, self.weights, n_jobs)
        return self.weights.reshape(-1, self.input_len)[bmus]

    def quantization_error(self, data: np.ndarray, n_jobs: int = -1) -> float:
        """Mean euclidean distance between each sample and its best matching unit."""
        data = self._check_data(data)
        return float(np.linalg.norm(data - self.quantization(data, n_jobs), axis=1).mean())

    def filter_noisy_samples(self, data: np.ndarray, labels: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """Flags samples whose label disagrees with the majority label of their neuron (False = noisy)."""
        winners = self.predict(data, n_jobs)
        labels = np.asarray(labels)
        _, label_ids = np.unique(labels, return_inverse=True)
        n_labels = int(label_ids.max()) + 1 if len(label_ids) else 0
        counts = np.zeros((self.x * self.y, n_labels), dtype=np.int64)
        np.add.at(counts, (winners, label_ids), 1)
        # argmax keeps the smallest label on ties, like np.unique + argmax did.
        majority = counts.argmax(axis=1)
        return label_ids == majority[winners]
