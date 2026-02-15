"""Tests for the UMAP embedding service."""

import numpy as np

from server.services.embeddings import UMAPProjector


def test_umap_2d_projection() -> None:
    """UMAP projects to correct 2D shape."""
    projector = UMAPProjector()
    vectors = [[float(i + j) for j in range(8)] for i in range(20)]
    result = projector.fit_transform(vectors, n_components=2)
    assert isinstance(result, np.ndarray)
    assert result.shape == (20, 2)


def test_umap_3d_projection() -> None:
    """UMAP projects to correct 3D shape."""
    projector = UMAPProjector()
    vectors = [[float(i + j) for j in range(8)] for i in range(20)]
    result = projector.fit_transform(vectors, n_components=3)
    assert result.shape == (20, 3)


def test_umap_caching() -> None:
    """Same input returns cached result."""
    projector = UMAPProjector()
    vectors = [[float(i + j) for j in range(4)] for i in range(15)]
    result1 = projector.fit_transform(vectors, n_components=2)
    result2 = projector.fit_transform(vectors, n_components=2)
    np.testing.assert_array_equal(result1, result2)


def test_umap_small_n() -> None:
    """UMAP handles small N by reducing n_neighbors."""
    projector = UMAPProjector()
    # Need at least 5 points for UMAP's spectral embedding to avoid scipy eigsh errors
    vectors = [[float(i + j) for j in range(4)] for i in range(5)]
    result = projector.fit_transform(vectors, n_components=2)
    assert result.shape == (5, 2)


def test_umap_different_dims_different_cache() -> None:
    """2D and 3D projections of same data are cached separately."""
    projector = UMAPProjector()
    vectors = [[float(i + j) for j in range(8)] for i in range(20)]
    result_2d = projector.fit_transform(vectors, n_components=2)
    result_3d = projector.fit_transform(vectors, n_components=3)
    assert result_2d.shape != result_3d.shape
