"""Data preparation modules: vectors, texts, KG chunks, KG datasets (NER/RE/ER), GGUF models.

Exports the PrepTask ABC and all task registries for unified status reporting.
"""

from benchmarks.harness.prep.base import PrepTask
from benchmarks.harness.prep.gguf_models import GGUF_PREP_TASKS, GGUFModelPrepTask
from benchmarks.harness.prep.kg_chunks import KGChunksPrepTask, chunks_prep_tasks
from benchmarks.harness.prep.kg_datasets import KG_PREP_TASKS, CrossNERPrepTask, FebrlPrepTask, HFDatasetPrepTask
from benchmarks.harness.prep.texts import TEXT_PREP_TASKS, GutenbergTextPrepTask
from benchmarks.harness.prep.vectors import VECTOR_PREP_TASKS, VectorPrepTask

__all__ = [
    "PrepTask",
    "GutenbergTextPrepTask",
    "TEXT_PREP_TASKS",
    "VectorPrepTask",
    "VECTOR_PREP_TASKS",
    "KGChunksPrepTask",
    "chunks_prep_tasks",
    "CrossNERPrepTask",
    "HFDatasetPrepTask",
    "FebrlPrepTask",
    "KG_PREP_TASKS",
    "GGUFModelPrepTask",
    "GGUF_PREP_TASKS",
]
