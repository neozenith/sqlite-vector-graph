"""UMAP projection service with caching."""

import hashlib
import logging

import numpy as np
import umap  # type: ignore[import-untyped]

log = logging.getLogger(__name__)

# Module-level projector singleton
_projector: "UMAPProjector | None" = None


class UMAPProjector:
    """Cached UMAP projector for embedding visualization."""

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def _cache_key(self, vectors: list[list[float]], n_components: int) -> str:
        """Generate a cache key from vector data shape and hash."""
        arr = np.array(vectors, dtype=np.float32)
        data_hash = hashlib.md5(arr.tobytes()).hexdigest()[:12]  # noqa: S324
        return f"{arr.shape[0]}x{arr.shape[1]}_{n_components}_{data_hash}"

    def fit_transform(self, vectors: list[list[float]], n_components: int = 2) -> np.ndarray:
        """Project high-dimensional vectors to 2D/3D via UMAP with caching."""
        key = self._cache_key(vectors, n_components)

        if key in self._cache:
            log.debug("UMAP cache hit: %s", key)
            return self._cache[key]

        arr = np.array(vectors, dtype=np.float32)
        n_samples = arr.shape[0]

        # UMAP n_neighbors must be < n_samples
        n_neighbors = min(15, max(2, n_samples - 1))

        log.info(
            "UMAP projecting %d vectors (%d-D -> %d-D, n_neighbors=%d)",
            n_samples,
            arr.shape[1],
            n_components,
            n_neighbors,
        )

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        projected = reducer.fit_transform(arr)

        self._cache[key] = projected
        return projected  # type: ignore[no-any-return]


def get_projector() -> UMAPProjector:
    """Get or create the singleton UMAP projector."""
    global _projector
    if _projector is None:
        _projector = UMAPProjector()
    return _projector
