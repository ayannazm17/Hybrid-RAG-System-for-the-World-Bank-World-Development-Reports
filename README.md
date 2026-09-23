# A Hybrid RAG System for the World Bank World Development Reports

Retrieval-augmented generation system built over the World Bank's World Development Report (WDR) series (2016–2025) — ten flagship annual reports, each 300+ pages, spanning distinct global development themes from *Digital Dividends* (2016) to *Middle Class: Life on the Ladder* (2025).

## Why RAG here

Three reasons this domain needs retrieval rather than prompting alone:

1. **Scale** — the ten PDFs run to several thousand pages combined, far past any LLM's context window.
2. **Recency gap** — standard LLM training data has partial-to-no coverage of the 2024–2025 reports.
3. **Verifiability** — the target readership (policymakers, development economists, NGOs) needs page-attributed, checkable answers, not confident generalisation.

Corpus sourced as ten official PDFs from the World Bank's Open Knowledge Repository, parsed entirely locally with no live web calls during the pipeline run.

## Architecture

**Baseline pipeline:** dense FAISS search only → top-5 chunks → generation.

**Enhanced pipeline**, three additions on top of the baseline:

- **Hybrid retrieval (RRF):** dense search (FAISS, inner-product index, `all-MiniLM-L6-v2` embeddings) fused with sparse BM25Okapi keyword search via Reciprocal Rank Fusion (k=60) — captures both semantic similarity and exact-match signals.
- **Cross-encoder reranking:** RRF's top-20 candidates re-scored jointly by `cross-encoder/ms-marco-MiniLM-L-6-v2`, cut down to a final top-5.
- **HyDE (Hypothetical Document Embeddings):** the query is first expanded by Gemini 3.1 Flash Lite into a short hypothetical passage a WDR would plausibly contain, and *that* is used as the retrieval query — pulls the search vector into the corpus's distributional space rather than the raw question's.

## Pipeline detail

- **Parsing:** PyMuPDF, page by page; near-blank pages (<100 chars) discarded
- **Chunking:** sliding window, 600 words, 100-word overlap
- **Caching:** parsed pages, chunks, embeddings and indices all cached to disk (`diskcache`, pickle, `.npy`, `.index`) so the expensive parsing step runs once
- **Embedding:** batches of 256, `all-MiniLM-L6-v2` (384-dim), normalised, indexed in FAISS `IndexFlatIP`
- **Generation:** both pipelines inject the top-5 chunks as `[WDR year | Page X]`-tagged context blocks into a structured system prompt requiring answers to be grounded only in that context, with citations. The enhanced pipeline additionally requires structured output (direct answer / supporting evidence with citations / caveats).

## Evaluation

10-query test set across four categories: simple factual recall, deep cross-document synthesis, ambiguous broad questions, and out-of-scope edge cases. Scored by Gemini 3.1 Flash Lite (LLM-as-judge) on Correctness, Grounding and Completeness (1–5).

| Metric | Baseline | Enhanced | Δ |
|---|---|---|---|
| Correctness | 4.30 | 4.70 | +0.40 |
| Grounding | 3.70 | 3.50 | −0.20 |
| Completeness | 4.80 | 4.90 | +0.10 |
| **Average** | **4.267** | **4.368** | **+0.101** |

## What worked, what didn't

- **Q1 (year-specific factual query, WDR 2016):** the clearest win for the enhanced pipeline. Baseline retrieved chunks from 2018/2020/2021/2025 and correctly — but uselessly — returned "no information about WDR 2016." Enhanced retrieved five correct WDR 2016 chunks and gave a complete, grounded answer. Aggregate scores understate how large this qualitative gap actually is.
- **Q4 (cross-document synthesis, 2016–2021 evolution):** the one case where enhanced scored *lower* (4.0 vs baseline's 4.67). HyDE generated a hypothetical passage anchored in 2022 financial-crisis language, pulling retrieval toward the wrong report year — a direct illustration of HyDE's core limitation: synthetic-passage quality inherits the generating LLM's own knowledge biases.
- **Citation hallucination**, the main recurring failure mode across *both* pipelines: the model cites plausible-looking page numbers that aren't actually in the retrieved context (e.g. baseline citing "page 63" for a governance query, when no retrieved chunk contained that page). This is instruction-following pressure — told to cite pages, the model invents one when the context doesn't supply it.

## Proposed improvements

- Semantic paragraph-boundary chunking instead of sliding-window, to stop tables/narrative text bleeding into each other
- Metadata pre-filtering by year when the query names one explicitly (would have fixed Q1's failure mode at near-zero cost, and prevents HyDE's Q4-style year-drift)
- Self-consistency reranking: multiple HyDE hypotheses, averaged embeddings, to reduce variance from a single stochastic expansion
- Multimodal OCR pass — PyMuPDF currently garbles or skips figures/tables/maps, losing real analytical content
- Async batching / paid API tier — free-tier Gemini rate limits (15 req/min) made the full eval run take ~10 minutes

## Repo contents

- Source doc — full write-up (domain rationale, architecture, implementation, evaluation, analysis)
- `evaluation_results.csv` — per-query LLM-as-judge scores (referenced in report; not included here)
