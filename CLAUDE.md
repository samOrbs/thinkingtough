# Project Rules — Toughness Training RAG Chatbot

## Canvas Diagrams — ALWAYS UPDATE
Whenever you make changes to the codebase, you MUST update the relevant Obsidian canvas diagrams:
1. **`obsidian-vault/System Architecture.canvas`** — full system architecture (aspirational + what's built)
2. **`obsidian-vault/Implementation Plan.canvas`** — build plan with sequential steps
3. **`obsidian-vault/Work Log.canvas`** — record of what was actually done, issues, fixes

If a change affects the architecture, update System Architecture.
If a change affects the build plan, update Implementation Plan.
ALWAYS update the Work Log with what was done.

## Safety Rules
- **NEVER use `rm -rf`** on user directories — NTFS is case-insensitive in WSL2, Recycle Bin is bypassed
- **Cost check before API calls** — calculate and confirm cost before running embedding/generation batches
- **Always `git commit`** after completing a phase or significant change

## Technical Notes
- SDK: `google.genai` (NOT deprecated `google.generativeai`)
- Embeddings: `gemini-embedding-001` (3072 dims)
- BM25 pickle: store chunks as plain dicts (not Chunk objects) to avoid module path issues
- Skip pages 1-15 and 218-232 when indexing (front/back matter)
- ChromaDB metadata: lists must be stringified
