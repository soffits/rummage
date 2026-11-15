---
name: rummage
description: Search the user's own local files (notes, documents, PDFs, pictures, chats) by meaning or exact words with the rummage CLI, then open the passages it finds. Use when the answer may be in the user's files and you don't know where, or the exact words.
---

# rummage

`rummage` keeps a local index of the user's files and finds things by what they mean and by the exact words in
them. It runs on the user's machine; nothing is sent anywhere.

## Search

```bash
rummage search "what the user is looking for, in their own words" --json -n 10
rummage search "INV-20931" --words-only --json          # names, IDs, numbers: exact words, no model, instant
rummage search "..." --json --path notes/work --glob '*.md' -s notes   # narrow by folder, pattern, source
rummage search "..." --json --since 2026-10-01 --until 2026-10-31      # a time window (dates you work out)
rummage search "Ana Souza" --json --sort newest -n 30                  # the newest matches first
rummage search "..." --json --budget-tokens 1500                       # shorter output
```

The JSON has `status`:

- `ok`: `results` are ranked. `summary.strong` says how many score near the top; the rest are weak.
- `no_results`: nothing matched. Rephrase by meaning, try exact words with `--words-only`, or drop a scope
  filter.
- `empty_index` or an error with status `index_missing`: nothing is indexed yet (`rummage index`).

Each result has `id`, `source`, `title`, `ref` (where it is), `lines` for text files, `score`, `snippet`.
`next_actions` holds the commands that show what the output left out (masked credentials, the rest of a partially
indexed file); the next query is yours to choose.

## Open what you found

```bash
rummage get '#3f2a9c01be'            # the whole item's text
rummage get '#3f2a9c01be' --loc 1800  # only the matched passage (a result's `loc`)
rummage get 'notes/lease.md' --head 40
```

For a chat result, `rummage get '#id' --around 3` shows the turns before and after it; for a file passage it shows
the lines around it. `rummage recent --minutes 30 --json` lists what was said or changed lately, newest first.

Credentials in results print as `[secret]`. Add `--show-secrets` only when the user asks for a value.

Quote the file and lines you relied on. A miss doesn't prove the thing isn't there: try other words, exact
names, or the folder directly before saying so.

While the index is still being built, output carries an `index` note saying how much is in; treat a miss then as
"maybe not indexed yet", and `rummage status --line` tells the user how far along it is.

## When rummage itself is broken

If you've confirmed a problem is rummage's own (reproduced it, checked `rummage doctor` and the version, ruled out
the user's config), tell the user and offer a ready issue draft: steps to reproduce, rummage version, OS, the exact
error, with private content (file names, paths, text from their files) removed. Link
https://github.com/soffits/rummage/issues and let them decide whether to file it; never open issues or PRs yourself.
