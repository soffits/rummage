# rummage

Local search by meaning and exact words over your files, on your own machine. An embedding model (EmbeddingGemma 2 by
default) runs on the CPU through llama.cpp's `llama-server`, started as a child process for one run, or behind an
OpenAI-compatible endpoint; the index is one SQLite file. Licence: AGPL-3.0-or-later
(up to 0.2.x it was MIT); third-party notices in `NOTICE`.
User docs: `README.md`. Design and the reasons behind it: `docs/design.md`, `docs/decisions.md`. Known leftovers:
`docs/queue.md`.

## Layout

- `src/rummage/models.py` and `profiles/*.toml`: model profiles (files or endpoint, prompts, dimensions, version,
  thresholds, reference vectors). `runtime.py`: the pinned llama.cpp build, the `llama-server` child and the
  endpoint embedder. `switch.py`: a model switch's record and phases; `calibrate.py`: per-model thresholds;
  `compare.py`: `rummage models compare`.
- `store.py` (SQLite schema), `text.py` (normalisation, CJK tokens, chunking), `search.py` (ranking), `indexer.py`
  (one run, per-item commits, lock), `cli.py`, `daemon.py` (the query daemon), `extract.py` (file bytes to text).
- `documents.py`: one file's bytes to segments for every source of files (text sniffing, documents, notebooks,
  pictures and scanned pages with OCR from `ocr.py`, recordings) and why a file kept only its name;
  `legacy_office.py` (.doc/.xls/.ppt) and `iwork.py` (Pages, Keynote, Numbers) are its format readers.
- `sources/`: the source base and entry-point discovery (`rummage.sources` group), the core types `markdown` and `files`
  (folder walking in `folder.py`). `plugins/<name>/`: service-specific source types (seafile, dsh-sessions, hermes,
  telegram-stickers), each registered as an entry point in `pyproject.toml`, with its own extra when it needs packages.
  `chats.py` is the shared turn model for chat plugins.
- A source type's name and its items' keys and fingerprints are the index format: changing them re-embeds everything.

## Rules for this project

- Sources are read-only: open files read-only (SQLite with `mode=ro&immutable=1`), never write next to what is indexed,
  keep downloaded bytes in memory.
- Anything that changes a built-in model's vectors (model file, quantization, dimension, prompt format, llama.cpp build
  that fails the reference check) changes the `version` in its profile; indexes then switch in the background.
- New dependencies go in an optional extra unless every install needs them; the core needs only numpy and pypdf.
- Text in the repo is English. CJK appears only where it tests or implements CJK handling.

## Checks and releases

- `uv run --with pytest --extra archives --extra dsh-sessions --extra watch --extra office --extra pdf-scans pytest -q` runs
  the tests (no model needed); `uv run --with pytest --extra ocr pytest -q tests/test_ocr.py` runs real OCR (downloads
  its models once). CI runs both on pushes to `main` and on pull requests.
- A real run needs the model: `rummage setup`, then `rummage index` and `rummage search` against a small config.
- Release: update `CHANGELOG.md`, bump `version` in `pyproject.toml` and `src/rummage/__init__.py`, merge to `main`,
  then push the tag `vX.Y.Z`; the `install` workflow then installs it on each system and runs a real index and search.
- Scope: rummage reads, finds, opens and says what it left out; judgment (next query, time phrases, duplicates,
  contradictions, people) is the calling agent's. Reasons in `docs/decisions.md`, "Scope".

## Working on the project

- Everything in the repo (code, docs, commits) is English, except translations and files deliberately in another language.
- Ordinary implementation choices are yours: decide, write the choice and its one-line reason under the right system group of `docs/decisions.md` (replacing any line it supersedes, no dates), and mention it in your report. Product decisions (what it indexes, direction, scope, taste) belong to the maintainers and live in `docs/roadmap.md` only; ask about those.
- If a requested approach looks weaker than an alternative, say so with the reason before building it.
- Report each state separately: implemented, tested (which checks, which real run), committed, pushed, CI. Say plainly what was not verified.
- Commit on a branch as you go. Merge to `main`, tag, release or publish only when the task asks for it.
- Work only in this checkout, on the current branch. Make no folders or worktrees outside it; builds people test come from the build's own output folder, given by path. Temporary files go under the build folder or a git-ignored path and are removed when done.

## Delegating to subagents

Do the work yourself, in your own session, by default: design, code, fixes and review. A subagent starts cold, re-reads what you already know and hands back a summary you must check, so it usually costs more than it saves. Hand off only work that is simple and would burn your context for little (a mechanical sweep over many files, a long log to search, a bulk rename), to a smaller, cheaper model that fits the job.

- A brief says what and why, not how: the goal, the decisions that apply, the non-obvious constraints, which files it owns, the checks that decide "done", and a progress file to write early and update after each step. Leave out methods nobody chose, so a capable model can find a better way.
- When you do hand off, parallel subagents only on disjoint files. Each shared helper has one owner; tell the others to use it or wait for it.
- A subagent's summary is a claim. Read its diff and rerun the checks before building on it.
- If a review is asked for, the reviewer is at least as strong as the writer (a weaker reviewer makes strong code worse) and marks each finding *likely* (hits ordinary use: data loss, crash, wrong result) or *rare* (hostile input, odd timing, limits nobody reaches). Fix the likely ones with a regression test; put rare ones in `docs/queue.md` unless the fix is cheap and local.

## Long work

- Any run can be cut off (limits, restarts, context compaction). For work longer than a few steps, keep `docs/progress/<task>.md`: the goal, what is done, decisions, what is next. Update it after each meaningful step so a fresh session resumes without redoing work. End it with `## Final` (what was done, checked, left over). It is a working note: when the work merges, move what lasts into the decisions and queue docs and delete the file.
- Finish the whole goal in one run. Don't end your turn to "wait" for something while work remains you can do.
- Leftovers and known bugs go into `docs/queue.md` the moment you find them, not into a promise in your report.

## Before building

- Before writing any non-trivial component, search the community and current research for maintained libraries and better approaches. Check the licence (MIT/Apache/BSD/CC0 to ship; GPL, non-commercial and unlicensed work only as ideas, reimplemented), upkeep, and that it runs on our versions. Adopt, adapt or write our own, and record why in `docs/decisions.md`.
- Take versions of languages, tools, frameworks and dependencies from current official sources, not memory; they change faster than training data.

## Code

- No backward compatibility while the project has no users: when a format or name changes, change it everywhere and delete the old one.
- Few general mechanisms, no special cases. When something breaks, fix the shared layer or the data, not the one call site. No hard-coded lists of names, no matching on error text, hostnames or model names as a substitute for a real signal.
- Define each list, limit or constant once and derive variants from it.
- Reproduce a bug before fixing it; a confident guess from reading code is often wrong. If two fixes built on one assumption fail, the assumption is the bug.
- CI is where tests run. Locally, compile and run only a quick check you need while writing; push the branch and let CI run the suites.
- Keep CI time low. CI runs on pull requests and `main`, not on every branch push. One release is one branch and one pull request (no stacked PRs per feature). Commit locally as often as you like; push and open the PR when the first sizeable part compiles, keep working on the same branch, and push the next batch when ready (a newer push cancels the older run). Don't sit watching a run (no sleep-and-poll loops): keep working and check its result between steps; when nothing but the CI result is left, say so in your report and end the turn. Never wait on a test run: if a local run is truly needed (a failure CI can't explain), run it in the background, keep writing code, and fix what it reports when it ends. Write each check once, at the cheapest level that can see it (unit test with fake data or a fake server first; a run with the real model only for what needs it), and extend an existing test before adding one that repeats its setup. Docs-only changes start no run (`paths-ignore` on both triggers), but GitHub judges a pull request by its whole diff, so push docs and progress notes together with the next code batch, never alone to a PR that has code.
- If a test can't pass, explain why in the progress file. Never loosen or move an assertion to make it green.
- Split a file by responsibility before it grows past about 500 lines (generated code and static data excepted).
- No dead code, commented-out code or bare `TODO`; unfinished work is in `docs/queue.md`.
- README and other public docs are for users. Notes for maintainers and agents go here or in `docs/`.

## Git

- Conventional Commits, one-line subjects, a body only when the why isn't obvious. The repo's local author identity is set on purpose; commits use it as configured, since rewriting authors later means rewriting history.
- Squash merges by default. Whoever merges a branch deletes it, local and remote, in the same step (`gh pr merge --squash --delete-branch`), removes its worktree and runs `git fetch --prune`; only `main` and the current release branch stay.
- PR titles are Conventional Commit subjects (`feat: ...`, `fix(scope): ...`), because a squash merge makes the title the commit on the default branch; CI checks it.
- In GitHub Actions, reference well-known actions by major tag (`@v5`), not by commit SHA.
- Releases: move the `Unreleased` part of `CHANGELOG.md` under the new version's heading (English), bump the version, squash-merge to `main` with a Conventional Commit subject (`feat: <summary> (vX.Y.Z)`), and once CI is green on that commit push the tag `vX.Y.Z`; the release workflow builds and publishes with that section as the description, without rerunning tests. Don't name a working branch exactly like the tag: `git push origin vX.Y.Z` then fails as ambiguous (use `refs/tags/vX.Y.Z`).
- Versions follow semver, and the bump comes from the commit types since the last tag, not from feel: below 1.0, `feat` or a breaking change (`!` / `BREAKING CHANGE`) is a minor bump and `fix`/`perf` a patch; from 1.0, breaking is major, `feat` minor, `fix` patch; docs/ci/chore/test alone need no release. The release check (`.github/workflows/release-check.yml`, `.github/scripts/release-bump-check.py`) fails a tag whose bump is too small; run the script with the new version before tagging.
