# Changelog

## Unreleased

- Every result has a stable short id (`#3f2a9c01be`); `rummage get <#id|ref|source:key>` prints the item's text, one passage with `--loc`, or the first lines with `--head`, and asks remotes when the item isn't local (`GET /get` on `serve`).
- Unchanged passages keep their vectors: an edited file only embeds what changed, and a moved file embeds nothing, by matching a hash of each passage's model input (or picture bytes) against stored vectors.

## 0.3.0

- Relicensed from MIT to AGPL-3.0-or-later; 0.2.x and earlier stay MIT. Third-party notices in `NOTICE`.
- Sources are plugins: types are discovered through the `rummage.sources` entry-point group. The core ships `markdown` and the new `files` source (text, HTML, PDF, Office and OpenDocument files, pictures, archive listings); `seafile`, `dsh-sessions`, `hermes` and `telegram-stickers` are bundled plugins with their own extras. Type names and the index format are unchanged.
- `rummage sources` lists the installed source types.
- PDF text is part of the core (pypdf); extras are now `archives`, `seafile` and `dsh-sessions`.

## 0.2.0

- Index time budget (`index --max-minutes`), quiet `doctor`, proxy for remotes, and `llama-server` stops with its parent.
- `dsh-sessions`: bot sends recorded in a topic's session are attributed to the assistant.
- `dsh-sessions`: the sender and channel come from the message's `source.principal` when it has one, before the text tag or OIDC notice; a Telegram prompt with a principal and no rpcId now counts as a person's message.

## 0.1.0

- First release: markdown, telegram-stickers, hermes, dsh-sessions and seafile sources; hybrid search; HTTP API; Docker image.
