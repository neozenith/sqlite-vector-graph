# Tokenisation, Chunking, and Multilingual Challenges

Captured: 2026-02-16. Context: knowledge graph pipeline design for muninn.

## Why Chunking Is Required

LLMs have finite context windows (4K-128K tokens). Raw documents exceed this.
More importantly, entity/relation extraction **quality degrades on long inputs** — the model's attention dilutes. Microsoft GraphRAG defaults to 300 tokens per chunk for this reason.

Chunks use sliding windows with 10-20% overlap so entities straddling boundaries aren't lost.

## Chunking Depends on the Tokenizer

Embedding models have a **max token window** (not word window):

| Model | Max Tokens | Tokenizer |
|-------|-----------|-----------|
| all-MiniLM-L6-v2 | 256 | WordPiece |
| all-mpnet-base-v2 | 384 | BPE |
| text-embedding-3-small (OpenAI) | 8,191 | cl100k BPE |
| nomic-embed-text-v1.5 | 8,192 | SentencePiece |

If a chunk exceeds the model's token limit, it **silently truncates**. The chunker ideally counts tokens the same way the downstream model does. Word-count approximation works for English (~1.3 BPE tokens/word) but breaks for other languages.

## BPE (Byte Pair Encoding) — How It Works

1. Start with bytes/characters
2. Count adjacent pairs across a training corpus
3. Merge the most frequent pair into a new symbol
4. Repeat thousands of times

Output: a **merge table** (ordered pair-merge rules) + **vocabulary** (token-to-integer mapping). At inference, the tokenizer replays merges greedily on new text.

BPE was originally a data compression algorithm (Gage, 1994). Sennrich et al. (2016) adapted it for neural MT.

## The Tokenization Tax — Multilingual Disparity

BPE operates on bytes so it *works* on all languages. But vocabularies are trained on English-heavy corpora, so non-English text fragments into many more tokens per semantic unit:

| Language | Tokens per English-equivalent | Why |
|----------|------------------------------|-----|
| English | 1.0x (baseline) | Dominant in training data |
| German, French | 1.2-1.5x | Shared Latin script helps |
| Chinese | 2-3x | No spaces, each character may split |
| Japanese | 2-4x | Three scripts, no spaces |
| Hindi, Arabic | 3-5x | Root morphology, missing vowels |
| Finnish, Turkish | 2-3x | Agglutinative (one word = whole phrase) |
| Myanmar, Khmer | 5-10x | No spaces, small training corpora |

### Impact on Chunking

- A 300-word chunk in English ~ 400 tokens
- A 300-word chunk in Japanese ~ 800-1200 tokens (overflows 512-token models)
- Word-count chunkers silently produce oversized chunks for non-English text

## Difficult Language Categories

### No Word Boundaries (no spaces)
- **Japanese** — 3 scripts (kanji, hiragana, katakana) intermixed
- **Chinese (Mandarin/Cantonese)** — each character can be a word or part of a compound
- **Thai, Khmer, Myanmar/Burmese, Lao** — continuous script, no spaces

### Agglutinative Morphology (arbitrarily long words)
- **Finnish** — `epäjärjestelmällistyttämättömyydellänsäkäänköhän` is one valid word
- **Turkish** — `evlerinizden` = "from your houses" = 4 morphemes stacked
- **Hungarian, Estonian, Korean, Swahili, Quechua** — similar patterns

### Script and Directionality
- **Arabic** — RTL, 3-consonant root system, vowels usually omitted
- **Hebrew** — RTL, missing vowels
- **Hindi/Sanskrit** — compound words can be very long, conjunct consonants
- **Georgian** — unique script, 4 types of verbal agreement

## What Libraries Exist Under the Hood

| Library | Core Language | C API? | License | Notes |
|---------|--------------|--------|---------|-------|
| tiktoken (OpenAI) | **Rust** | No | MIT | PyO3 bindings, no C FFI |
| sentencepiece (Google) | **C++** | No | Apache-2.0 | Protobuf dependency, heavyweight |
| HF tokenizers | **Rust** | No (declined) | Apache-2.0 | Issue #185 closed NOT_PLANNED |
| llama.cpp tokenizer | **C++** | Partial | MIT | BPE+SPM+WPM+UGM, coupled to GGUF |
| tokenizers-cpp (MLC-AI) | C++/Rust | **Yes** | Apache-2.0 | Compiles Rust to `libtokenizers_c.a` |
| utf8proc | **C** | **Yes** | MIT | Unicode normalization only, 2 files |

**No production-quality pure-C BPE library exists.** This is a gap in the ecosystem.

## Implications for muninn

### Current State
Chunking is in Python (`kg_extract.py:chunk_fixed_tokens`) using whitespace-split word counting. This is English-only and approximate.

### Options for a C TVF

1. **Simple `text_chunk()` TVF** (~200 lines C11, no deps): recursive character splitting on `\n\n` -> `\n` -> `. ` -> ` `, with configurable max size. Language-agnostic if sized by character count.

2. **Character-count mode**: Chunk by character count (not word count) with a `chars_per_token` parameter. ~4 chars/token for English, ~1-2 for CJK. Avoids needing BPE while preventing silent truncation.

3. **Minimal BPE in C** (~300 lines): load a pre-trained merge table, apply greedy BPE merges. Tractable but requires shipping vocabulary files.

4. **Link tokenizers-cpp**: full HF tokenizer compatibility but introduces Rust build dependency.

### Recommendation
A character-count-based `text_chunk()` TVF with sentence-boundary awareness covers 95% of the value. **Sentence-boundary-aware splitting matters more than token-accurate counting** for KG extraction quality. Token-accurate BPE is a future enhancement if multilingual support becomes a priority.

## References

- Gage (1994) — "A New Algorithm for Data Compression" (original BPE)
- Sennrich et al. (2016) — "Neural Machine Translation of Rare Words with Subword Units" (BPE for NLP)
- Petrov et al. (2023) — "Language Model Tokenizers Introduce Unfairness Between Languages" (tokenization tax)
- Microsoft GraphRAG — default 300-token chunking with glean step
- LangChain `RecursiveCharacterTextSplitter`, LlamaIndex `SentenceSplitter`
