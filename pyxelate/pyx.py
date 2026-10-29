import warnings
from typing import Optional, Union, Tuple

import numpy as np
from scipy.ndimage import convolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.extmath import randomized_svd

from .fast_ops import (
    resize_float,
    rgb2hsv_fast,
    hsv2rgb_fast,
    rgb2oklab,
    oklab_palette_labels,
    equalize_adapthist_fast,
    sobel_downsample_numba,
    floyd_dither_numba,
    median_hsv_fast,
    dilate_rgb_fast,
)
from .pal import BasePalette
from .styles import apply_style, STYLE_FUNCTIONS


class PyxWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class BGM(BayesianGaussianMixture):
    """Wrapper for BayesianGaussianMixture with fast OKLab perceptual palette initialization."""

    MAX_ITER = 128
    RANDOM_STATE = 1234567

    def __init__(self, palette: Union[int, BasePalette], find_palette: bool) -> None:
        self.palette = palette
        self.find_palette = find_palette
        if self.find_palette:
            super().__init__(
                n_components=self.palette,
                max_iter=self.MAX_ITER,
                covariance_type="tied",
                weight_concentration_prior_type="dirichlet_distribution",
                weight_concentration_prior=1.0 / self.palette,
                mean_precision_prior=1.0 / 256.0,
                warm_start=False,
                random_state=self.RANDOM_STATE,
            )
        else:
            super().__init__(
                n_components=len(self.palette),
                max_iter=self.MAX_ITER,
                covariance_type="tied",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1e-7,
                mean_precision_prior=1.0 / len(self.palette),
                warm_start=False,
                random_state=self.RANDOM_STATE,
            )
            self.mean_prior = np.mean([val[0] for val in self.palette], axis=0)

    def _initialize_parameters(self, X: np.ndarray, random_state: int, xp=None) -> None:
        """Initializes centroids using K-Means or fast perceptual OKLab distance."""
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        if self.find_palette:
            label = (
                KMeans(n_clusters=self.n_components, n_init=1, random_state=random_state)
                .fit(X)
                .labels_
            )
        else:
            pal_arr = np.array([p[0] for p in self.palette], dtype=np.float64)
            label = oklab_palette_labels(X, pal_arr)
        resp[np.arange(n_samples), label] = 1
        self._initialize(X, resp)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BGM":
        converged = True
        with warnings.catch_warnings(record=True) as w:
            super().fit(X)
            if w and w[-1].category == ConvergenceWarning:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                converged = False
        if not converged:
            warnings.warn(
                "Pyxelate could not properly assign colors, try a different palette size for better results!",
                PyxWarning,
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = super().predict_proba(X)
        if self.find_palette:
            if self.palette < 3:
                return np.sqrt(p)
        elif len(self.palette) < 3:
            return np.sqrt(p)
        return p


class Pyx(BaseEstimator, TransformerMixin):
    """High-performance Numba-accelerated Pyx transformer with 12 built-in visual styles."""

    BGM_RESIZE = 192
    SCALE_RGB = 1.07
    HIST_BRIGHTNESS = 1.19
    COLOR_QUANT = 8
    DITHER_NAIVE_BOOST = 1.33
    SVD_N_COMPONENTS = 32
    SVD_MAX_ITER = 8
    SVD_RANDOM_STATE = 1234
    DITHER_BAYER_MATRIX = np.array(
        [
            [-0.5, 0.0, -0.375, 0.125],
            [0.25, -0.25, 0.375, -0.125],
            [-0.3125, 0.1875, -0.4375, 0.0625],
            [0.4375, -0.0625, 0.3125, -0.1875],
        ]
    )

    def __init__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        factor: Optional[int] = None,
        upscale: Union[Tuple[int, int], int] = 1,
        depth: int = 1,
        palette: Union[int, BasePalette] = 8,
        dither: Optional[str] = "none",
        sobel: int = 3,
        svd: bool = False,
        alpha: float = 0.6,
        style: Optional[str] = None,
    ) -> None:
        if (width is not None or height is not None) and factor is not None:
            raise ValueError(
                "You can only set either height + width or the downscaling factor, but not both!"
            )
        self.height = int(height) if height else None
        self.width = int(width) if width else None
        self.factor = int(factor) if factor else None
        self.sobel = max(2, int(sobel))
        if isinstance(upscale, (list, tuple, set, np.ndarray)):
            self.upscale = (int(upscale[0]), int(upscale[1]))
        else:
            self.upscale = (int(upscale), int(upscale))
        self.depth = max(1, int(depth))
        self.palette = palette
        self.find_palette = isinstance(self.palette, (int, float))
        assert dither in (
            None,
            "none",
            "naive",
            "bayer",
            "floyd",
            "atkinson",
        ), "Unknown dithering algorithm!"
        self.dither = dither
        self.svd = bool(svd)
        self.alpha = float(alpha)
        if style is not None and style not in STYLE_FUNCTIONS:
            raise ValueError(f"Unknown style '{style}'. Valid styles: {list(STYLE_FUNCTIONS)}")
        self.style = style
        self.model = BGM(self.palette, self.find_palette)
        self.is_fitted = style is not None
        self.palette_cache = None

    def _get_size(self, original_height: int, original_width: int) -> Tuple[int, int]:
        if self.height is not None and self.width is not None:
            return self.height, self.width
        elif self.height is not None:
            return self.height, max(1, int(self.height / original_height * original_width))
        elif self.width is not None:
            return max(1, int(self.width / original_width * original_height)), self.width
        elif self.factor is not None:
            return max(1, original_height // self.factor), max(1, original_width // self.factor)
        else:
            return original_height, original_width

    def _image_to_float(self, image: np.ndarray) -> np.ndarray:
        if np.issubdtype(image.dtype, np.integer):
            return np.clip(np.asarray(image, dtype=np.float64) / 255.0, 0.0, 1.0)
        return np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)

    def _image_to_int(self, image: np.ndarray) -> np.ndarray:
        if isinstance(image, BasePalette):
            image = np.array(image.value, dtype=np.float64)
        elif isinstance(image, (list, tuple)):
            image = np.array(image, dtype=np.float64)
        if image.dtype.kind == "f":
            return np.clip(image * 255.0, 0, 255).astype(int)
        return image

    @property
    def colors(self) -> np.ndarray:
        if self.palette_cache is None:
            if self.find_palette:
                assert self.is_fitted, "Call 'fit(image_as_numpy)' first!"
                c = rgb2hsv_fast(self.model.means_.reshape(-1, 1, 3))
                c[:, :, 1:] *= self.SCALE_RGB
                c = hsv2rgb_fast(c)
                c = np.clip(c * 255 // self.COLOR_QUANT * self.COLOR_QUANT, 0, 255).astype(int)
                c[c < self.COLOR_QUANT * 2] = 0
                c[c > 255 - self.COLOR_QUANT * 2] = 255
                self.palette_cache = c.reshape(-1, 3)
            else:
                self.palette_cache = self._image_to_int(self.palette).reshape(-1, 3)
        return self.palette_cache

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pyx":
        if self.style is not None:
            self.is_fitted = True
            return self
        h, w, d = X.shape
        X_ = resize_float(
            X[:, :, :3], min(h, self.BGM_RESIZE), min(w, self.BGM_RESIZE), anti_aliasing=False
        ).reshape(-1, 3)
        if self.find_palette:
            X_ = np.clip(((X_ - 0.5) * self.SCALE_RGB) + 0.5, 0.0, 1.0)
        self.model.fit(X_)
        self.palette_cache = None
        self.is_fitted = True
        return self

    def _pyxelate(self, X: np.ndarray) -> np.ndarray:
        """Numba-accelerated Sobel gradient-weighted downsampling across all 3 channels."""
        h, w, c = X.shape
        pad_h = (self.sobel - (h % self.sobel)) % self.sobel
        pad_w = (self.sobel - (w % self.sobel)) % self.sobel
        if pad_h > 0 or pad_w > 0:
            X_pad = np.pad(X, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        else:
            X_pad = np.ascontiguousarray(X, dtype=np.float64)
        return sobel_downsample_numba(X_pad, self.sobel)

    def _svd(self, X: np.ndarray) -> np.ndarray:
        if self.SVD_N_COMPONENTS >= X.shape[0] - 1 and self.SVD_N_COMPONENTS >= X.shape[1] - 1:
            return X
        out = np.empty_like(X)
        for ch in range(3):
            U, s, V = randomized_svd(
                X[:, :, ch],
                n_components=min(self.SVD_N_COMPONENTS, min(X.shape[:2]) - 1),
                n_iter=self.SVD_MAX_ITER,
                random_state=self.SVD_RANDOM_STATE,
            )
            out[:, :, ch] = np.clip(U.dot(np.diag(s).dot(V)), 0.0, 1.0)
        return out

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None, frame_index: int = 0
    ) -> np.ndarray:
        if self.style is not None:
            eff_factor = self.factor or 3
            eff_upscale = self.upscale[0] if self.upscale[0] > 1 else eff_factor
            return apply_style(
                X,
                style=self.style,
                factor=eff_factor,
                upscale=eff_upscale,
                frame_index=frame_index,
            )

        assert self.is_fitted, "Call 'fit(image_as_numpy)' first!"
        h, w, d = X.shape
        new_h, new_w = self._get_size(h, w)
        alpha_mask = None
        if d > 3:
            alpha_mask = resize_float(X[:, :, 3], new_h, new_w, anti_aliasing=True)

        if self.depth:
            new_h *= self.sobel ** self.depth
            new_w *= self.sobel ** self.depth

        X_ = resize_float(X[:, :, :3], new_h, new_w, anti_aliasing=True)
        if self.svd:
            X_ = self._svd(X_)

        X_ = rgb2hsv_fast(equalize_adapthist_fast(X_))
        X_[:, :, 1:] *= self.HIST_BRIGHTNESS
        X_ = hsv2rgb_fast(np.clip(X_, 0.0, 1.0))

        for _ in range(self.depth):
            if d == 3:
                X_ = median_hsv_fast(X_)
            X_ = self._pyxelate(X_)

        final_h, final_w, _ = X_.shape
        if self.find_palette:
            X_ = np.clip(((X_ - 0.5) * self.SCALE_RGB) + 0.5, 0.0, 1.0)
        reshaped = np.reshape(X_, (final_h * final_w, 3))

        if self.dither is None or self.dither == "none":
            probs = self.model.predict(reshaped)
            X_ = self.colors[probs]
        elif self.dither == "naive":
            probs = self.model.predict_proba(reshaped)
            p = np.argmax(probs, axis=1)
            X_ = self.colors[p].copy()
            probs[np.arange(len(p)), p] = 0
            p2 = np.argmax(probs, axis=1)
            v1 = np.max(probs, axis=1) > (1.0 / (len(self.colors) + 1))
            v2 = np.max(probs, axis=1) > (1.0 / (len(self.colors) * self.DITHER_NAIVE_BOOST + 1))
            pad = not bool(final_w % 2)
            for i in range(0, len(X_), 2):
                m = (i // final_w) % 2
                idx = i + m if (pad and i + m < len(X_)) else i
                if m and v1[idx]:
                    X_[idx] = self.colors[p2[idx]]
                elif not m and v2[idx]:
                    X_[idx] = self.colors[p2[idx]]
        elif self.dither == "bayer":
            probs = self.model.predict_proba(reshaped)
            tiled = np.tile(self.DITHER_BAYER_MATRIX, (final_h // 4 + 1, final_w // 4 + 1))[
                :final_h, :final_w
            ].reshape(-1, 1)
            idx = np.argmax(probs + tiled * 0.18, axis=1)
            X_ = self.colors[idx]
        elif self.dither == "floyd":
            probs = self.model.predict_proba(reshaped)
            probs = np.ascontiguousarray(
                [probs[:, i].reshape((final_h, final_w)) for i in range(len(self.colors))],
                dtype=np.float64,
            )
            res = floyd_dither_numba(probs, final_h, final_w)
            X_ = self.colors[res.reshape(final_h * final_w)]
        elif self.dither == "atkinson":
            probs = self.model.predict_proba(reshaped)
            probs = np.ascontiguousarray(
                [probs[:, i].reshape((final_h, final_w)) for i in range(len(self.colors))],
                dtype=np.float64,
            )
            res = floyd_dither_numba(probs, final_h, final_w)
            X_ = self.colors[res.reshape(final_h * final_w)]

        X_ = np.reshape(X_, (final_h, final_w, 3))
        if alpha_mask is not None:
            a_u8 = np.where(alpha_mask >= self.alpha, 255, 0).astype(int)
            X_ = np.dstack((X_[:, :, :3], a_u8))

        X_ = np.repeat(np.repeat(X_, self.upscale[0], axis=0), self.upscale[1], axis=1)
        return X_.astype(np.uint8)
