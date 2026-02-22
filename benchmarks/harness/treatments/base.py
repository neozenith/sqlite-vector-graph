"""Base Treatment ABC for all benchmark permutations.

A Treatment represents one specific benchmark configuration (one permutation).
The harness calls setup -> run -> teardown and collects metrics automatically.

Subclasses MUST implement:
    category        — e.g., 'vss', 'graph', 'adjacency', 'kg-extract'
    permutation_id  — unique human-readable slug used as folder name
    label           — human-readable description
    setup()         — create tables, load data
    run()           — execute benchmark, return metrics dict
    teardown()      — clean up

Subclasses MAY override:
    params_dict()   — return all parameters as flat dict for JSONL serialization
"""

import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ProgressTracker:
    """Time-throttled loop progress reporter.

    Logs progress updates (e.g. "  50/500 (10%)") only when at least
    `interval_s` seconds have elapsed since the last log.
    """

    def __init__(self, total: int, interval_s: float = 5.0, logger: logging.Logger | None = None) -> None:
        self._total = total
        self._interval_s = interval_s
        self._log = logger or log
        self._last_log_time = time.monotonic()

    def update(self, current: int) -> None:
        """Report progress if enough time has elapsed since the last log."""
        now = time.monotonic()
        if now - self._last_log_time >= self._interval_s:
            pct = (current / self._total * 100) if self._total > 0 else 0
            self._log.info("    %d/%d (%.0f%%)", current, self._total, pct)
            self._last_log_time = now


class Treatment(ABC):
    """Base class for all benchmark treatments."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Benchmark category: 'vss', 'graph', 'centrality', 'community',
        'graph_vt', 'kg-extract', 'kg-resolve', 'kg-graphrag', 'node2vec'."""

    @property
    @abstractmethod
    def permutation_id(self) -> str:
        """Human-readable slug, unique across all permutations.

        Used as folder name under results/ and as the --id value for CLI.
        Example: 'vss_muninn-hnsw_MiniLM_ag-news_n5000'
        """

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable description for manifest display.

        Example: 'VSS: muninn-hnsw / MiniLM / ag_news / N=5000'
        """

    @abstractmethod
    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        """Create tables, load data, prepare for benchmark.

        Returns a dict of setup_metrics (e.g., {'n_vectors_loaded': 5000}).
        """

    @abstractmethod
    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Execute the benchmark measurement.

        Returns a dict of treatment-specific metrics (e.g., {
            'insert_rate_vps': 12345.6,
            'search_latency_ms': 0.42,
            'recall_at_k': 0.95,
        }).
        """

    @abstractmethod
    def teardown(self, conn: sqlite3.Connection) -> None:
        """Clean up resources after benchmark."""

    def params_dict(self) -> dict[str, Any]:
        """Return all treatment parameters as a flat dict for JSONL serialization.

        Subclasses should override to include their specific parameters.
        The harness merges this with common metrics when writing JSONL.
        """
        return {}

    @property
    def requires_muninn(self) -> bool:
        """Whether the harness should load the muninn extension before setup().

        When True (default), the harness loads muninn and treats failure as fatal.
        When False, the harness skips loading — the treatment handles its own
        extensions (e.g., sqlite-vector, vectorlite, sqlite-vec).

        Subclasses that don't need muninn should override this to return False.
        """
        return True

    @property
    def sort_key(self) -> tuple[Any, ...]:
        """Return a tuple for sorting permutations by size (ascending).

        The first element should be the primary scaling dimension (the x-axis
        metric), followed by secondary dimensions for tie-breaking.
        Subclasses should override to reflect their scaling dimension.
        Default sorts by permutation_id.
        """
        return (self.permutation_id,)

    @contextmanager
    def step(self, name: str):
        """Context manager that logs entry/exit of a named step with elapsed time."""
        log.info("  [STEP] %s ...", name)
        t0 = time.perf_counter()
        try:
            yield
        except Exception:
            elapsed = time.perf_counter() - t0
            log.info("  [STEP] %s failed (%.1fs)", name, elapsed)
            raise
        else:
            elapsed = time.perf_counter() - t0
            log.info("  [STEP] %s done (%.1fs)", name, elapsed)

    def progress(self, total: int, interval_s: float = 5.0) -> ProgressTracker:
        """Create a time-throttled progress tracker for loop reporting."""
        return ProgressTracker(total, interval_s=interval_s, logger=log)
