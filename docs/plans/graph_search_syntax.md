# Graph Search Syntax — dbt-inspired Node Selection TVF

**Status:** Draft Plan
**Date:** 2026-02-16

## Motivation

dbt's node selection syntax is a powerful mini-language for selecting subsets of a DAG.
Bringing it to SQLite as a TVF would let users write concise, expressive graph queries
like "give me everything NOT reachable from this critical node" — which immediately
surfaces dead code paths, orphaned subgraphs, and dependency gaps.

The key insight: **node selection + set complement = dead code detection**.

```sql
-- Find all nodes NOT in the dependency tree of 'critical_node'
SELECT * FROM graph_select('edges', 'src', 'dst', 'not @critical_node');

-- Ancestors 2 levels up, descendants 3 levels down
SELECT * FROM graph_select('edges', 'src', 'dst', '2+build_step+3');
```

## Reference: dbt Node Selection Syntax

### Core Operators

| Operator | Position | Selects |
|----------|----------|---------|
| `node` | bare | Just the node itself |
| `+node` | prefix | Node + all ancestors (upstream) |
| `node+` | suffix | Node + all descendants (downstream) |
| `+node+` | both | Node + all ancestors + all descendants |
| `N+node` | prefix | Node + ancestors to depth N |
| `node+M` | suffix | Node + descendants to depth M |
| `N+node+M` | both | Ancestors to depth N, descendants to depth M |
| `@node` | prefix | Node + descendants + ancestors of ALL descendants |

### The `@` Operator (Transitive Build Closure)

This is the most powerful operator. Given:

```
A    B    X
 \  /    /
  C    Y
 / \  /
D    E
     |
     F
```

- `C+` → `{C, D, E, F}` — just descendants
- `+C` → `{A, B, C}` — just ancestors
- `+C+` → `{A, B, C, D, E, F}` — ancestors + descendants
- `@C` → `{A, B, C, D, E, F, X, Y}` — descendants + **all ancestors of those descendants**

The `@` operator captures the full "build closure" — everything needed to rebuild
the subgraph rooted at the selected node. This is critical for CI pipelines:
if you modify `C`, you need to rebuild `E`, but `E` also depends on `Y` and `X`,
so you need those too.

### Set Operators

| Syntax | Operation | Example |
|--------|-----------|---------|
| `A B` | Union (OR) | `+model_a +model_b` — ancestors of both |
| `A,B` | Intersection (AND) | `+model_a,+model_b` — common ancestors only |
| `not A` | Complement | `not @critical` — everything NOT in the `@` set |
| `A not B` | Difference | `+root not tag:deprecated` — ancestors minus deprecated |

### Attribute Selectors (dbt's `method:value`)

dbt supports filtering by metadata: `tag:nightly`, `config.materialized:table`,
`path:models/staging`, etc. For muninn, we adapt this to column-based filtering
(see Design section).

### Wildcards

`*` matches any sequence of characters in node names:
- `stg_*` — all nodes starting with `stg_`
- `*.base.*` — nodes with `base` in path segments

### Composition Examples

```bash
# Everything 2 hops up and 3 hops down from build_step
2+build_step+3

# Union of two lineage trees
+orders +sessions

# Intersection: only nodes common to both ancestries
+orders,+sessions

# Full build closure of a node, minus deprecated
@critical_node not tag:deprecated

# Dead code: everything NOT reachable from any production path
not @production_entry_point

# Two hops downstream from all nodes tagged "source"
tag:source+2
```

## Design: `graph_select` TVF

### Proposed SQL Interface

```sql
-- Basic: topology-only selection
SELECT node FROM graph_select(
    'edge_table',       -- table containing edges
    'src_col',          -- source column name
    'dst_col',          -- destination column name
    'selector_expr'     -- the dbt-like selector expression
);

-- With attribute table for method:value selectors
SELECT node FROM graph_select(
    'edge_table',
    'src_col',
    'dst_col',
    'selector_expr',
    'node_table',       -- optional: table with node attributes
    'node_id_col'       -- optional: ID column in node_table
);
```

### Output Columns

| Column | Type | Description |
|--------|------|-------------|
| `node` | TEXT | The selected node ID |
| `depth` | INTEGER | Distance from the selector anchor (0 = the named node) |
| `direction` | TEXT | `'ancestor'`, `'descendant'`, or `'self'` |
| `selector` | TEXT | Which sub-expression matched this node |

### Architecture

```
graph_select_tvf.c  ── SQLite TVF glue (xConnect, xBestIndex, xFilter, etc.)
    ├── graph_selector_parse.c  ── Tokenizer + recursive-descent parser → AST
    ├── graph_selector_eval.c   ── AST evaluator → node set (using graph_load.c)
    └── graph_load.c            ── Existing shared graph loading infrastructure
```

### Parser: Selector Expression Grammar

```ebnf
expression     = term { SPACE term }             (* union *)
term           = "not" atom                       (* complement *)
               | atom { "," atom }                (* intersection *)
atom           = ["@"] depth_spec                 (* @ operator *)
depth_spec     = [INT "+"] selector ["+" [INT]]   (* depth-limited traversal *)
selector       = method ":" value                 (* attribute-based *)
               | value                            (* node name or glob *)
method         = "tag" | "attr" | "col"           (* extensible *)
value          = GLOB_PATTERN                     (* supports * and ? wildcards *)
```

Tokens:
- `SPACE` — whitespace (union separator)
- `,` — intersection separator
- `+` — ancestor/descendant operator
- `@` — build closure operator
- `not` — complement keyword
- `INT` — depth limit (digits)
- `:` — method/value separator
- `GLOB_PATTERN` — `[a-zA-Z0-9_*?.\-]+`

### Evaluator: Set Operations on Node IDs

The evaluator works in terms of **node sets** (bit vectors over the loaded graph's node indices):

```c
typedef struct {
    uint8_t *bits;       /* bit vector: 1 bit per node in GraphData */
    int capacity;        /* = GraphData.node_count, rounded up to byte boundary */
} NodeSet;

/* Set operations — all O(N/8) where N = total nodes */
void nodeset_union(NodeSet *dst, const NodeSet *a, const NodeSet *b);
void nodeset_intersect(NodeSet *dst, const NodeSet *a, const NodeSet *b);
void nodeset_complement(NodeSet *dst, const NodeSet *a, int total_nodes);
void nodeset_difference(NodeSet *dst, const NodeSet *a, const NodeSet *b);
```

Evaluation of each atom:
1. **Bare node** `my_node` → set containing just that node (look up via `graph_data_find`)
2. **Prefix `+`** → BFS/DFS backwards through `GraphData.in[]` (reverse adjacency)
3. **Suffix `+`** → BFS/DFS forwards through `GraphData.out[]` (forward adjacency)
4. **`@` operator** → forward BFS to get descendants, then for each descendant, backward BFS to get all ancestors
5. **`not`** → complement the result set against all nodes
6. **Depth limits** → cap BFS depth at N
7. **Wildcards** → iterate all node IDs, glob-match, union the matches
8. **Attribute selectors** → SQL query against the node attribute table, build set from results

### AST Node Types

```c
typedef enum {
    SEL_NODE,          /* literal node name or glob pattern */
    SEL_ATTR,          /* method:value attribute selector */
    SEL_ANCESTORS,     /* +node with optional depth */
    SEL_DESCENDANTS,   /* node+ with optional depth */
    SEL_BOTH,          /* +node+ or N+node+M */
    SEL_CLOSURE,       /* @node */
    SEL_UNION,         /* A B — two children */
    SEL_INTERSECT,     /* A,B — two children */
    SEL_COMPLEMENT,    /* not A — one child */
} SelectorType;

typedef struct SelectorNode {
    SelectorType type;
    char *value;                /* node name, glob, or attr value */
    char *method;               /* for SEL_ATTR: "tag", "col", etc. */
    int depth_up;               /* ancestor depth limit, -1 = unlimited */
    int depth_down;             /* descendant depth limit, -1 = unlimited */
    struct SelectorNode *left;  /* first child (or only child for complement) */
    struct SelectorNode *right; /* second child (for union/intersect) */
} SelectorNode;
```

### Leveraging Existing Infrastructure

The muninn codebase already has everything needed for the graph operations:

| Need | Already Have | Location |
|------|-------------|----------|
| Load graph from any table | `graph_data_load()` | `graph_load.c` |
| Forward adjacency (descendants) | `GraphData.out[]` | `graph_load.h` |
| Reverse adjacency (ancestors) | `GraphData.in[]` | `graph_load.h` |
| O(1) node lookup by name | `graph_data_find()` | `graph_load.c` |
| SQL identifier validation | `id_validate()` | `id_validate.c` |
| BFS traversal logic | BFS in `graph_tvf.c` | `graph_tvf.c` |
| Eponymous TVF pattern | All existing TVFs | `graph_tvf.c` |

What's **new**:
- Selector expression parser (tokenizer + recursive descent)
- Bit-vector node set with set operations
- AST evaluator that maps selector semantics to graph operations
- Optional attribute table integration

### Attribute Selectors for SQLite

dbt's `tag:nightly` assumes metadata is in the dbt manifest. For generic SQLite
graphs, we adapt this to column-based lookups:

```sql
-- If 'nodes' table has columns: id, type, status, department
-- Then col:type=source selects all nodes where type='source'

SELECT node FROM graph_select(
    'edges', 'src', 'dst',
    'col:type=source+2',     -- descendants 2 deep from all source-type nodes
    'nodes', 'id'            -- attribute table and its ID column
);
```

Supported attribute methods:

| Method | Syntax | Meaning |
|--------|--------|---------|
| `col:name=value` | Column equals value | `col:status=active` |
| `col:name=pat*` | Column glob match | `col:type=stg_*` |

This keeps it simple and generic — any column in a node attributes table
can be used as a filter.

## Implementation Plan

### Phase 1: Core Parser + Topology Operators

**Files to create:**
- `src/graph_selector_parse.h` — parser API
- `src/graph_selector_parse.c` — tokenizer + recursive-descent parser
- `src/graph_selector_eval.h` — evaluator API + NodeSet type
- `src/graph_selector_eval.c` — AST evaluation + set operations
- `src/graph_select_tvf.c` — SQLite TVF wrapper
- `src/graph_select_tvf.h` — registration function
- `test/test_graph_selector.c` — C unit tests for parser + evaluator

**Files to modify:**
- `src/muninn.c` — register the new TVF
- `Makefile` — add new source files to build

**Scope:**
- Parse: `node`, `+node`, `node+`, `+node+`, `N+node+M`, `@node`
- Set ops: space (union), comma (intersection), `not` (complement)
- No attribute selectors yet
- No wildcards yet

**Estimated complexity:** ~800-1000 lines of C (parser ~300, eval ~300, TVF ~200, tests ~200)

### Phase 2: Wildcards + Attribute Selectors

- Add glob matching on node IDs (`*`, `?`)
- Add `col:name=value` attribute selectors with optional node table parameter
- Add `selector` output column showing which expression matched

### Phase 3: Advanced Features

- `--exclude` equivalent (second TVF parameter for exclusion expression)
- Performance optimization: lazy evaluation, short-circuit on intersection
- Optional `weight_col` support for weighted depth limits
- Optional `direction` parameter (default "forward", matching existing TVFs)

## Example Queries (Aspirational)

```sql
-- Dead code detection: find nodes not reachable from production
SELECT node FROM graph_select('deps', 'src', 'dst', 'not @main_pipeline');

-- Impact analysis: what breaks if I change this node?
SELECT node FROM graph_select('deps', 'src', 'dst', '@changed_module');

-- Narrow scope: 2 hops up, 3 hops down
SELECT node FROM graph_select('deps', 'src', 'dst', '2+my_service+3');

-- Common ancestors of two leaf nodes
SELECT node FROM graph_select('deps', 'src', 'dst', '+leaf_a,+leaf_b');

-- Everything downstream of sources, minus deprecated
SELECT node FROM graph_select('deps', 'src', 'dst',
    'col:type=source+ not col:status=deprecated', 'nodes', 'id');

-- Union of two critical paths
SELECT node FROM graph_select('deps', 'src', 'dst',
    '@api_gateway @data_pipeline');
```

## Risks & Considerations

### Selector Ambiguity

The `+` character appears in both the depth operator and potentially in node names.
Mitigation: node names follow `id_validate()` rules (`[a-zA-Z0-9_]` only), so `+`
is unambiguous as an operator.

### Performance on Large Graphs

The `@` operator is O(V + E) for the descendant BFS, then O(V + E) again for each
ancestor BFS. On very large graphs (millions of nodes), this could be slow.
Mitigation: the graph is already fully loaded into memory by `graph_data_load()`;
bit-vector set operations are O(V/8) which is fast.

### Complement Set Size

`not @X` on a large graph returns potentially millions of nodes. This is inherent
to the semantics — the caller asked for the complement. Could add a `LIMIT` hint
in the TVF, or let SQLite's normal `LIMIT` clause handle it.

### Parser Complexity

The grammar is simple enough for a hand-written recursive-descent parser (no need
for a parser generator). The dbt source uses regex, but C recursive descent is
actually cleaner for this grammar's complexity level.

### Existing BFS Reuse

The BFS logic in `graph_tvf.c` is tightly coupled to the traversal TVF cursor model.
For `graph_select`, we need a simpler BFS that just fills a NodeSet bit vector.
This is a ~30-line function, not worth trying to share with the existing BFS.

## Comparison: TVF vs Scalar Function

| Approach | Pros | Cons |
|----------|------|------|
| **TVF** (`graph_select`) | Returns rows, composable with JOIN/WHERE, natural for sets | More boilerplate (vtab API) |
| **Scalar** (`graph_match`) | Simpler to implement, returns JSON array | Not composable, can't join results |
| **Both** | Maximum flexibility | Double the maintenance |

**Recommendation:** TVF. It fits the existing pattern, is composable with SQL,
and returning a result set is the natural representation of a node selection.

## References

- [dbt Node Selection Syntax](https://docs.getdbt.com/reference/node-selection/syntax)
- [dbt Graph Operators](https://docs.getdbt.com/reference/node-selection/graph-operators)
- [dbt Set Operators](https://docs.getdbt.com/reference/node-selection/set-operators)
- [dbt selector_spec.py source](https://github.com/dbt-labs/dbt-core/blob/main/core/dbt/graph/selector_spec.py)
