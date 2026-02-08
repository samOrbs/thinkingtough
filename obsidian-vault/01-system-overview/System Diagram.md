# Book Expert RAG — System Architecture

> A production-grade conversational RAG system that makes an LLM a deep expert on a scanned book, with persistent memory and agentic capabilities.

## Master Pipeline Diagram

```mermaid
flowchart TB
    subgraph PHASE1["PHASE 1: DOCUMENT PROCESSING (One-Time)"]
        direction TB
        A1[/"232 JPG Page Images"/] --> B1["OCR / Transcription<br/><b>Gemini 2.0 Flash</b>"]
        B1 --> C1["Text Cleaning<br/>& Normalization"]
        C1 --> D1["Structure Extraction<br/>Chapters, Sections, Headings"]
        D1 --> E1["Quality Validation<br/>Confidence Scoring"]
        E1 --> F1[/"Clean Markdown<br/>with Page Metadata"/]
    end

    subgraph PHASE2["PHASE 2: INDEXING PIPELINE (One-Time)"]
        direction TB
        F1 --> G1["Semantic Chunking<br/>~512 tokens, 100 overlap"]
        G1 --> G2["Contextual Enrichment<br/><b>Anthropic Contextual Retrieval</b>"]

        G2 --> H1["Embedding Generation<br/><b>Voyage-4-large</b>"]
        G2 --> H2["BM25 Index<br/>Keyword Search"]

        H1 --> I1[("ChromaDB<br/>Vector Store")]
        H2 --> I2[("BM25 Index<br/>Sparse Store")]

        G1 --> J1["RAPTOR Summaries<br/>Tree of Abstractions"]
        J1 --> I1
    end

    subgraph PHASE3["PHASE 3: RETRIEVAL PIPELINE (Per Query)"]
        direction TB
        K1[/"User Query"/] --> L1["Query Understanding<br/>& Classification"]

        L1 --> M1["Multi-Query Expansion<br/>3 query variants"]
        L1 --> M2["HyDE<br/>Hypothetical Answer"]

        M1 --> N1["Vector Search<br/>Top-50 candidates"]
        M1 --> N2["BM25 Search<br/>Top-50 candidates"]
        M2 --> N1

        N1 --> O1["Reciprocal Rank Fusion<br/>Merge & Deduplicate"]
        N2 --> O1

        O1 --> P1["Reranking<br/><b>Cohere Rerank 3.5</b><br/>Top-50 → Top-5"]
        P1 --> Q1["Context Assembly<br/>Ordered by relevance"]
    end

    subgraph PHASE4["PHASE 4: GENERATION (Per Query)"]
        direction TB
        Q1 --> R1["System Prompt<br/>Book Expert Persona<br/>+ Book Summary (2K tokens)"]
        R1 --> R2["Model Router<br/>Simple → Flash<br/>Complex → Sonnet"]
        R2 --> S1["LLM Generation<br/><b>Gemini Flash / Claude Sonnet</b>"]
        S1 --> T1["Response Validation<br/>Citation Check<br/>Grounding Check"]
        T1 --> U1[/"Answer with<br/>Page Citations"/]
    end

    subgraph PHASE5["PHASE 5: MEMORY & PERSISTENCE"]
        direction LR
        V1[("SQLite<br/>Conversation<br/>History")]
        V2[("Fact Store<br/>Extracted<br/>Knowledge")]
        V3[("User Notes<br/>& Plans")]
        V4["Session Manager<br/>Summarize Old Turns<br/>Load Relevant Context"]
    end

    subgraph PHASE6["PHASE 6: AGENTIC LAYER"]
        direction LR
        W1["search_book()"]
        W2["create_note()"]
        W3["search_notes()"]
        W4["create_plan()"]
        W5["get_page()"]
        W6["summarize_chapter()"]
        W7["create_flashcards()"]
    end

    U1 --> V1
    V4 --> R1
    PHASE6 --> PHASE3
    PHASE6 --> PHASE5

    style PHASE1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style PHASE2 fill:#1a1a2e,stroke:#f59e0b,color:#fff
    style PHASE3 fill:#1a1a2e,stroke:#3b82f6,color:#fff
    style PHASE4 fill:#1a1a2e,stroke:#10b981,color:#fff
    style PHASE5 fill:#1a1a2e,stroke:#8b5cf6,color:#fff
    style PHASE6 fill:#1a1a2e,stroke:#ec4899,color:#fff
```

## Detailed Subsystem Diagrams

### Document Processing Detail

```mermaid
flowchart LR
    subgraph OCR["OCR Pipeline"]
        direction TB
        IMG[/"page_0001.jpg<br/>through<br/>page_0232.jpg"/]
        IMG --> BATCH["Gemini Batch API<br/>50 pages per batch"]
        BATCH --> PROMPT["Transcription Prompt<br/>Preserve structure<br/>Output markdown"]
        PROMPT --> RAW[/"Raw Markdown<br/>per page"/]
    end

    subgraph CLEAN["Cleaning Pipeline"]
        direction TB
        RAW --> DEHYPH["Rejoin Hyphenated<br/>Line-Break Words"]
        DEHYPH --> HEADER["Remove Running<br/>Headers & Footers"]
        HEADER --> NORM["Unicode<br/>Normalization"]
        NORM --> STRUCT["Detect Headings<br/>Chapters, Sections"]
    end

    subgraph VALIDATE["Validation"]
        direction TB
        STRUCT --> SPELL["Spell Check<br/>Flag Anomalies"]
        SPELL --> CONF["Confidence Score<br/>Per Page"]
        CONF --> REVIEW["Human Review<br/>Low-Confidence Pages"]
        REVIEW --> FINAL[/"Final Markdown<br/>book.md"/]
    end
```

### Retrieval Detail

```mermaid
flowchart TB
    Q[/"User: What does Loehr say<br/>about recovery markers?"/]

    Q --> QU["Query Understanding"]
    QU --> |"Factual question<br/>about specific concept"| MQ["Multi-Query Generation"]

    MQ --> Q1["'recovery markers in<br/>toughness training'"]
    MQ --> Q2["'physiological indicators<br/>of athletic recovery'"]
    MQ --> Q3["'Loehr recovery<br/>measurement methods'"]

    Q1 & Q2 & Q3 --> VS["Vector Search<br/>(parallel)"]
    Q1 & Q2 & Q3 --> BS["BM25 Search<br/>(parallel)"]

    VS --> |"50 candidates"| RRF["Reciprocal Rank<br/>Fusion"]
    BS --> |"50 candidates"| RRF

    RRF --> |"~80 unique chunks"| RR["Cohere Rerank 3.5"]
    RR --> |"Top 5 chunks"| CA["Context Assembly"]

    CA --> |"Ordered by<br/>relevance, not<br/>position (avoid<br/>lost-in-middle)"| LLM["LLM Generation"]
```

### Memory Architecture Detail

```mermaid
flowchart TB
    subgraph SESSION["Per-Session Context"]
        direction LR
        SYS["System Prompt<br/>+ Book Summary<br/>(cached)"]
        RECENT["Last 10 Turns<br/>(verbatim)"]
        SUMMARY["Older Turns<br/>(compressed summary)"]
    end

    subgraph PERSIST["Persistent Storage"]
        direction LR
        CONV[("conversations<br/>table")]
        FACTS[("facts<br/>table")]
        NOTES[("user_notes<br/>table")]
        PLANS[("plans<br/>table")]
    end

    subgraph WARMUP["Session Warmup"]
        direction TB
        NEW["New Session Starts"] --> LOAD["Load Book Summary"]
        LOAD --> RELEVANT["Retrieve Relevant<br/>Past Facts & Notes"]
        RELEVANT --> INJECT["Inject into<br/>System Prompt"]
    end

    SESSION --> PERSIST
    WARMUP --> SESSION
```

## Navigation

| Section | Link |
|---------|------|
| Document Processing | [[02-document-processing/Overview]] |
| Indexing Pipeline | [[03-indexing-pipeline/Overview]] |
| Retrieval Pipeline | [[04-retrieval-pipeline/Overview]] |
| Generation Layer | [[05-generation-layer/Overview]] |
| Memory & Persistence | [[06-memory-persistence/Overview]] |
| Agentic Layer | [[07-agentic-layer/Overview]] |
| Evaluation & Testing | [[08-evaluation-testing/Overview]] |
| Deployment | [[09-deployment/Overview]] |

## Recommended Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| OCR | Gemini 2.0 Flash | Best accuracy/cost for printed text |
| Chunking | Semantic (paragraph-boundary) | Preserves natural text units |
| Enrichment | Anthropic Contextual Retrieval | 67% fewer retrieval failures |
| Embeddings | Voyage-4-large | SOTA MTEB scores |
| Vector DB | ChromaDB | Free, local, sufficient for 1 book |
| Keyword Search | rank_bm25 | Lightweight, no server needed |
| Reranking | Cohere Rerank 3.5 | Best quality/cost reranker |
| Daily LLM | Gemini 2.5 Flash | Cheapest capable model |
| Deep Analysis LLM | Claude Sonnet 4.5 | Best reasoning |
| Memory | SQLite + custom | Simple, portable, sufficient |
| Agent Framework | Claude tool use (no framework) | Minimal complexity |
| UI | Chainlit | Purpose-built chat UI for RAG |
| Evaluation | RAGAS + DeepEval | Industry standard metrics |

## Cost Summary

| Phase | One-Time | Monthly (~50 queries/day) |
|-------|----------|--------------------------|
| OCR Transcription | $0.07 | — |
| Contextual Enrichment | $1–3 | — |
| Embeddings | $0.03 | — |
| Vector DB | Free | Free |
| LLM Queries (Flash) | — | $5 |
| Reranking | — | $3 |
| Memory Storage | Free | Free |
| **Total** | **~$4** | **~$8/mo** |
