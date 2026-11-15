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

- One z-score scale per modality, not per script: scaling Chinese passages separately cut their drift into unrelated Chinese queries only slightly and pushed a Chinese answer down for a Chinese question on a mostly English corpus.

## Sources

- Source types are entry points in the `rummage.sources` group, so services are plugins and third parties can add their own; the core keeps only generic folder sources.
- Bundled plugins live in `rummage.plugins.*` with extras for their packages, rather than separate distributions: one install and one image, while the core never imports them.
- `dsh-sessions` reads who wrote a message from `source.principal` first (DSH's field for the caller behind a user message, filled by dsh-oidc and dsh-telegram-bot as `{kind, channel, id, name?}`), then from the text tag or OIDC notice: structured data beats parsing, and old logs have only the text.
- Chats: one item per message a person wrote plus the reply after it; the model reads the start of both, keyword search the full text.
- Seafile: an account token with read-only shares; head commit per library, then file object ids; archives listed over HTTP range requests.

## Licence

- AGPL-3.0-or-later from 0.3.0: `serve` makes it a network service, and modified hosted versions should share their source.
