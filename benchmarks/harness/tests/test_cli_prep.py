"""Tests for the prep subcommand CLI structure."""

from benchmarks.harness.tests.conftest import run_cli


class TestPrepCLI:
    def test_prep_help(self):
        """prep --help should list sub-subcommands."""
        result = run_cli("prep", "--help")
        assert result.returncode == 0
        assert "vectors" in result.stdout
        assert "texts" in result.stdout
        assert "kg-chunks" in result.stdout
        assert "kg" in result.stdout
        assert "gguf" in result.stdout
        assert "all" in result.stdout

    def test_prep_no_subcommand_shows_usage(self):
        """prep without a target should show usage and exit with error."""
        result = run_cli("prep")
        assert result.returncode != 0

    def test_prep_vectors_help(self):
        """prep vectors --help should show --status and --force flags."""
        result = run_cli("prep", "vectors", "--help")
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--model" in result.stdout
        assert "--dataset" in result.stdout

    def test_prep_texts_help_has_examples(self):
        """prep texts --help should show examples."""
        result = run_cli("prep", "texts", "--help")
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--book-id" in result.stdout
        assert "--random" in result.stdout
        assert "--category" in result.stdout
        assert "--list" in result.stdout
        assert "Examples:" in result.stdout

    def test_prep_kg_chunks_help(self):
        """prep kg-chunks --help should show --status and --force."""
        result = run_cli("prep", "kg-chunks", "--help")
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--book-id" in result.stdout

    def test_prep_kg_help(self):
        """prep kg --help should show --status, --force, and --dataset."""
        result = run_cli("prep", "kg", "--help")
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--dataset" in result.stdout

    def test_prep_all_help(self):
        """prep all --help should show --status and --force."""
        result = run_cli("prep", "all", "--help")
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout

    def test_prep_vectors_status(self):
        """prep vectors --status should show cache status table."""
        result = run_cli("prep", "vectors", "--status")
        assert result.returncode == 0
        assert "Vector Cache Status" in result.stdout

    def test_prep_texts_status(self):
        """prep texts --status should show text cache status."""
        result = run_cli("prep", "texts", "--status")
        assert result.returncode == 0
        assert "Text Cache Status" in result.stdout

    def test_prep_texts_list(self):
        """prep texts --list should show cached texts."""
        result = run_cli("prep", "texts", "--list")
        assert result.returncode == 0
        # Should show either cached texts or "No cached texts"
        assert "Cached Gutenberg Texts" in result.stdout or "No cached texts" in result.stdout

    def test_prep_kg_chunks_status(self):
        """prep kg-chunks --status should show chunk DB status."""
        result = run_cli("prep", "kg-chunks", "--status")
        assert result.returncode == 0
        assert "KG Chunk Database Status" in result.stdout

    def test_prep_kg_status(self):
        """prep kg --status should show KG dataset status."""
        result = run_cli("prep", "kg", "--status", timeout=60)
        assert result.returncode == 0
        assert "KG Dataset Status" in result.stdout

    def test_prep_gguf_help(self):
        """prep gguf --help should show --status, --force, and --model."""
        result = run_cli("prep", "gguf", "--help")
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--model" in result.stdout

    def test_prep_gguf_status(self):
        """prep gguf --status should show GGUF model status."""
        result = run_cli("prep", "gguf", "--status")
        assert result.returncode == 0
        assert "GGUF Model Status" in result.stdout

    def test_prep_all_status(self):
        """prep all --status should show status for all targets."""
        result = run_cli("prep", "all", "--status", timeout=60)
        assert result.returncode == 0
        assert "Vector Cache Status" in result.stdout
        assert "Text Cache Status" in result.stdout
        assert "KG Chunk Database Status" in result.stdout
        assert "KG Dataset Status" in result.stdout
        assert "GGUF Model Status" in result.stdout
