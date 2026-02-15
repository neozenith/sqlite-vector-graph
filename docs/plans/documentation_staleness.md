# Documentation Staleness Tracker

Inventory of documentation that was stale or incomplete following the addition of centrality, community detection (Leiden), and Node2Vec subsystems to the muninn extension.

**Status:** **Resolved.** Updated 2026-02-15. All items addressed.

**Extracted from:** [`knowledge_graph_benchmark.md`](knowledge_graph_benchmark.md) (Gap 2).

---

## Summary

| Document | Status | Resolution |
|----------|--------|------------|
| `docs/plans/graph_algorithms.md` | **Resolved** | File deleted (algorithms are implemented) |
| `docs/index.md` | **Resolved** | All 5 subsystems listed; installation section and guide links added |
| `mkdocs.yml` nav | **Resolved** | Added Getting Started, Guides (Centrality & Community, Node2Vec, GraphRAG Cookbook), reorganized Benchmarks |
| `mkdocs.yml` site_description | **Resolved** | Updated to include centrality + community detection |
| `README.md` | **Resolved** | Already had all 5 subsystems in features, diagram, API, SQL examples, and references |
| `CLAUDE.md` | **Resolved** | Already listed all 5 subsystems in architecture section |
| `skills/muninn/SKILL.md` | **Resolved** | Already says "Five subsystems" with correct TVF table |

---

## 1. README.md — Resolved

All sections were already up-to-date:
- Features list includes centrality + community
- Mermaid diagram shows all 5 subsystem boxes
- Quick Start SQL has betweenness + leiden examples
- API Reference has all TVF sections
- Research References includes Brandes (2001) and Traag (2019)

## 2. mkdocs Site — Resolved

New pages created:

| Page | File | Priority | Status |
|------|------|----------|--------|
| **API Reference** | `docs/api.md` | P1 | Already existed |
| **Getting Started / Installation** | `docs/getting-started.md` | P1 | **Created** |
| **Centrality & Community Guide** | `docs/centrality-community.md` | P2 | **Created** |
| **Node2Vec Guide** | `docs/node2vec.md` | P2 | **Created** |
| **GraphRAG Cookbook** | `docs/graphrag-cookbook.md` | P2 | **Created** |
| **Examples** | — | P2 | Deferred — examples linked from Getting Started |
| **Competitive Landscape** | — | P3 | Deferred — internal plan doc at `docs/plans/competitive_landscape.md` |

Updated `mkdocs.yml` nav to include all new pages with Guides section.

## 3. docs/index.md — Resolved

Added:
- Installation section with Python/Node.js/source tabs
- "Learn More" section with links to all guide pages

## 4. Plan Documents — Resolved

`docs/plans/graph_algorithms.md` was already deleted (algorithms are implemented).

## 5. CLAUDE.md — Resolved

Already correctly shows 5 subsystems in both the entry point section and module layering diagram.

## 6. Skills — Resolved

`skills/muninn/SKILL.md` already says "Five subsystems" with correct TVF table listing all 9 TVFs.
