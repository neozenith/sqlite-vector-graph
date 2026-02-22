"""Tests for the benchmark subcommand + subprocess isolation."""

from benchmarks.harness.tests.conftest import run_cli


class TestBenchmarkCLI:
    def test_unknown_id_exits_with_error(self):
        """An invalid permutation ID should exit with error."""
        result = run_cli("benchmark", "--id", "nonexistent_id_12345")
        assert result.returncode != 0

    def test_missing_id_flag_errors(self):
        """Omitting --id should show usage error."""
        result = run_cli("benchmark")
        assert result.returncode != 0

    def test_help_flag(self):
        """--help should work."""
        result = run_cli("benchmark", "--help")
        assert result.returncode == 0
        assert "--id" in result.stdout
        assert "--force" in result.stdout
