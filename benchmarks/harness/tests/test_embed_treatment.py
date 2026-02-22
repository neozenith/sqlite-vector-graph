"""Tests for the Embed treatment."""

import pytest

from benchmarks.harness.common import (
    EMBED_FNS,
    EMBED_SEARCH_BACKENDS,
    EMBED_SIZES,
    EMBEDDING_MODELS,
)
from benchmarks.harness.treatments.embed import EmbedTreatment

EMBED_FN_SLUGS = [e["slug"] for e in EMBED_FNS]
BACKEND_SLUGS = [b["slug"] for b in EMBED_SEARCH_BACKENDS]


class TestEmbedTreatmentInstantiation:
    def test_create_muninn_hnsw(self):
        t = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 500)
        assert t.category == "embed"
        assert "muninn_embed" in t.permutation_id
        assert "muninn-hnsw" in t.permutation_id
        assert "MiniLM" in t.permutation_id
        assert "ag-news" in t.permutation_id
        assert "n500" in t.permutation_id

    def test_create_lembed_pq(self):
        t = EmbedTreatment("lembed", "sqlite-vector-pq", "NomicEmbed", 768, "wealth_of_nations", 1000)
        assert t.category == "embed"
        assert "lembed" in t.permutation_id
        assert "sqlite-vector-pq" in t.permutation_id
        assert "NomicEmbed" in t.permutation_id
        assert "n1000" in t.permutation_id

    def test_create_sqlite_vec_brute(self):
        t = EmbedTreatment("muninn_embed", "sqlite-vec-brute", "BGE-Large", 1024, "ag_news", 100)
        assert t.category == "embed"
        assert "sqlite-vec-brute" in t.permutation_id

    def test_label_is_descriptive(self):
        t = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 500)
        assert "muninn_embed" in t.label
        assert "muninn-hnsw" in t.label
        assert "MiniLM" in t.label
        assert "500" in t.label

    def test_dataset_underscores_replaced(self):
        """Permutation IDs should use hyphens for dataset names."""
        t = EmbedTreatment("lembed", "muninn-hnsw", "MiniLM", 384, "wealth_of_nations", 500)
        assert "wealth-of-nations" in t.permutation_id
        assert "wealth_of_nations" not in t.permutation_id

    def test_permutation_id_format(self):
        """Permutation ID should follow the embed_{fn}+{backend}_{model}_{dataset}_n{N} pattern."""
        t = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 500)
        assert t.permutation_id == "embed_muninn_embed+muninn-hnsw_MiniLM_ag-news_n500"

    def test_permutation_id_contains_plus(self):
        """The + separator distinguishes embed_fn from search_backend in the ID."""
        t = EmbedTreatment("lembed", "sqlite-vector-pq", "MiniLM", 384, "ag_news", 100)
        assert "lembed+sqlite-vector-pq" in t.permutation_id


class TestEmbedTreatmentParams:
    def test_params_dict_keys(self):
        t = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 500)
        params = t.params_dict()
        assert params["embed_fn"] == "muninn_embed"
        assert params["search_backend"] == "muninn-hnsw"
        assert params["model"] == "MiniLM"
        assert params["dim"] == 384
        assert params["dataset"] == "ag_news"
        assert params["n"] == 500
        assert params["k"] == 10

    def test_params_dict_lembed(self):
        t = EmbedTreatment("lembed", "sqlite-vec-brute", "NomicEmbed", 768, "wealth_of_nations", 1000)
        params = t.params_dict()
        assert params["embed_fn"] == "lembed"
        assert params["search_backend"] == "sqlite-vec-brute"
        assert params["model"] == "NomicEmbed"
        assert params["dim"] == 768


class TestEmbedTreatmentSortKey:
    def test_sort_key_primary_is_n(self):
        """Primary sort dimension should be N (ascending)."""
        t100 = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 100)
        t500 = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 500)
        assert t100.sort_key < t500.sort_key

    def test_sort_key_secondary_is_dim(self):
        """Secondary sort should be by dimension."""
        t384 = EmbedTreatment("muninn_embed", "muninn-hnsw", "MiniLM", 384, "ag_news", 500)
        t768 = EmbedTreatment("muninn_embed", "muninn-hnsw", "NomicEmbed", 768, "ag_news", 500)
        assert t384.sort_key < t768.sort_key


class TestEmbedPermutationMatrix:
    @pytest.mark.parametrize("embed_fn", EMBED_FN_SLUGS)
    @pytest.mark.parametrize("backend", BACKEND_SLUGS)
    def test_all_fn_backend_combos_create(self, embed_fn, backend):
        """Every embed_fn x search_backend combo should create a valid treatment."""
        t = EmbedTreatment(embed_fn, backend, "MiniLM", 384, "ag_news", 100)
        assert t.category == "embed"
        assert embed_fn in t.permutation_id
        assert backend in t.permutation_id

    @pytest.mark.parametrize("model_name", list(EMBEDDING_MODELS.keys()))
    def test_all_models_create(self, model_name):
        """All embedding models should be valid for embed treatments."""
        dim = EMBEDDING_MODELS[model_name]["dim"]
        t = EmbedTreatment("muninn_embed", "muninn-hnsw", model_name, dim, "ag_news", 100)
        assert model_name in t.permutation_id

    def test_total_permutation_count(self):
        """Verify the expected total permutation count: 2 x 3 x 3 x 2 x 4 = 144."""
        n_fns = len(EMBED_FN_SLUGS)
        n_backends = len(BACKEND_SLUGS)
        n_models = len(EMBEDDING_MODELS)
        n_datasets = 2  # ag_news, wealth_of_nations
        n_sizes = len(EMBED_SIZES)
        expected = n_fns * n_backends * n_models * n_datasets * n_sizes
        assert expected == 144
