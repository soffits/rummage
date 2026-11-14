# Decisions

## Model and runtime

- EmbeddingGemma 2, first 512 of 768 dimensions, float16 storage: near-lossless on MTEB, a third less storage and compute.
- llama.cpp `llama-server` with ggml-org's Q8_0 GGUF: one runtime for text and images, official prebuilt CPU builds, the same files DigUp checked against the reference. ONNX int8 was ~25% faster on text but needs our own image preprocessing.
- The server runs as a child for one run on 127.0.0.1 and a free port; no resident daemon except under `rummage serve`.

## Index and ranking

- One SQLite file: items, segments with optional vectors, contentless FTS5, per-source cursors, run history.
- Ranking adapted from DigUp: z-scored cosine per modality, z ≥ 2 counts as a meaning match, word coverage bonus, ID and number queries weigh words double.
- CJK characters are single FTS tokens, so short Chinese words match; trigram tokenizers can't match two-character words.
- Brute-force numpy scoring instead of an ANN index or sqlite-vec: fast enough for a few hundred thousand vectors, no alpha dependency.

## Sources

- Chats: one item per message a person wrote plus the reply after it; the model reads the start of both, keyword search the full text.
- Seafile: an account token with read-only shares; head commit per library, then file object ids; archives listed over HTTP range requests.
