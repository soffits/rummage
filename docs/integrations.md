# Using rummage from agents

Any agent that can run a command can use rummage: `rummage search "<query>" --json` returns a versioned
`rummage.search.v1` object, and `rummage get <id>` opens a result. Two thin layers sit on top:

- **An Agent Skill** ([skills/rummage/SKILL.md](../skills/rummage/SKILL.md)) tells an agent when and how to search.
  Copy the folder into your agent's skills directory (for example `~/.claude/skills/` or `.agents/skills/`).
- **`rummage recall`** for prompt hooks: before each message reaches the model, the hook passes the prompt to
  recall, which returns a short block of cited passages when something in your files clearly matches, and nothing
  otherwise. It never loads the model on its own, so keep `rummage daemon` running for meaning matches; without it,
  recall matches exact words only. Each recalled line reads `#id · source · path:lines (or chat session#message) · date · "title" · why it matched`, followed by a snippet. Options: `--max-items` (4), `--max-chars` (1500), `--min-z` (3.0), `-s`,
  `--path`.
  Recall also adds up to two turns said in the last 45 minutes in other chat sessions (`--recent-minutes`,
  `--recent-items`) when the prompt shares words with them, so a conversation continued in a new window keeps its
  context; the current session is left
  out (`--session`, or the hook's `session_id`), and so are its own turns from that window, which the agent still has.
  A short prompt that leans on the turn before it ("the other one?", "rename it then") is searched by meaning with that turn
  in front of it: the hook's `context` or `--context` when given, else the session's newest turn in the index. A
  meaning match just above the gate counts only when it also shares some of the prompt's words.
  Credentials are masked in the block unless `--show-secrets`.
- **`rummage recent`** lists the latest chat turns and changed files across sources (`--minutes`, `-s`,
  `--exclude-session`, `--json`), and **`rummage get <#id> --around N`** shows a chat hit with the turns around it.

The adapters below call `rummage recall` before each prompt and add its block next to the user's message, never to
the system prompt, so the prompt cache keeps working. A recall that fails or takes longer than its timeout (5 s)
adds nothing and the turn goes on. Each one assumes `rummage` is on the agent's `PATH`.

## What a query sees: scopes

The index is one file, but what `search`, `recall` and `recent` see depends on the folder the query runs in
(`--cwd`, default the current folder; a hook's `cwd`):

- inside a git repository: that project. Its own files (a `local` source rooted there), the files of any other
  source that lie under it, and chat turns whose recorded working folder is in it (Claude Code, Codex, OpenCode,
  Gemini CLI and DeepSeek Harness sessions record one). Worktrees of one repository share its scope.
- inside a folder of a `[scopes.<name>]` table: the whole sources it lists (`sources = ["*"]` for all) plus what lies
  under its `dirs`. Use this for an assistant's home, which should see everything:

  ```toml
  [scopes.assistant]
  dirs = ["~/assistant"]
  sources = ["*"]
  ```

- anywhere else: `default_scope` (`"all"`, or a scope's name).

The deepest matching folder wins; a configured scope beats a project at the same folder. Sources marked
`local = true` (a project's own files) are seen only from inside their own folder. `--all` sees everything.
`search --json` and `recent --json` carry a `scope` object (`name`, `kind`: `project`, `scope`, `all` or
`everything`, `folder`, `sources`, `folders`, `hidden_sources`) and, when the scope left part of the index out, a
`widen_scope` entry in `next_actions` with the same command plus `--all`; text output ends with a `scope:` line. The
recall block says the scope in its last line when it is limited. `project_scopes = false` turns the git rule off.

## The prompt hook contract

Adapters need one process per message: `rummage recall --hook`, with the hook's JSON on stdin.

Input (unknown fields are ignored):

```json
{"prompt": "…", "session_id": "…", "cwd": "/path/the/agent/works/in", "harness": "opencode"}
```

`harness` is one of `claude-code`, `codex`, `opencode`, `gemini-cli`, `hermes`, `dsh`; `--harness` on the command
line does the same (the Claude Code and Codex hooks don't send it). With a `cwd`, the hook first does what
`rummage project ensure` does (below), then recalls within that folder's scope.

Output, always one JSON object, even when nothing matched:

```json
{
  "hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "Recalled from …"},
  "systemMessage": "rummage: recall paused until 16:30",
  "rummage": {
    "index": {"state": "complete", "line": "", "note": "", "…": "…"},
    "scope": {"name": "app", "kind": "project", "folder": "/work/app", "sources": [], "folders": ["/work/app"],
              "hidden_sources": []},
    "project": {"cwd": "/work/app/src", "project": {"folder": "/work/app", "source": "project-app-1a2b3c",
                "added": false}, "history": {"harness": "opencode", "source": "opencode", "state": "on"},
                "added": [], "started": false},
    "paused": {"id": "4f2a91", "what": "recall", "since": 1794769200, "until": 1794771000,
               "until_local": "2026-11-15 16:30", "here": "/work/app"}
  }
}
```

`hookSpecificOutput` is there only when there is a block for the model; `systemMessage` only when there is something
to tell the person: the pause while recall is paused, else the index status line while the index is incomplete.
`paused` is there only while a pause applies; `index` is described under
[Index progress for adapters](#index-progress-for-adapters). Adapters ignore fields they don't know.

## Projects without setup

`rummage project ensure [--cwd DIR] [--harness NAME] [--json]` makes the project a folder is in part of memory. It
is what the hook does with `cwd`, for adapters and people who want it separately, and it is safe to call on every
prompt: for a project already known it checks a few files and the config and returns (a few milliseconds, no
database work).

- In a git repository no configured source covers yet, it adds a source to the config: `type = "files"`, rooted at
  the repository, `git = true` (the files git tracks or would track, so `.gitignore` is respected), `local = true`,
  pictures off. Files that hold keys (`*.pem`, `id_rsa*`, `.env` and other dot files, `credentials*`…) are skipped.
  Every file whose content is text is read as text, whatever its extension, so source code and build files are
  found by what they say; it is project memory, not a code search engine (no symbols or call graphs).
- With `--harness`, it switches that harness's own session history on when it is on this machine and not configured
  yet: installing a harness's rummage plugin is the choice to have its chats remembered, so a conversation that
  started before the plugin still gets recall from the project's earlier chats. A history switched off in the config
  stays off. Other harnesses' histories stay off until `rummage sources enable`.
- When something was added, or no `rummage watch` runs for this data folder, it starts `rummage watch --setup --on-demand` in
  the background (which first fetches the model files when they are missing, then reads every enabled source and
  keeps them current). A running watch restarts itself when the config changes, so new sources are picked up.
  A watch started any other way (a service, a process supervisor, by hand) takes over from an on-demand one, so
  auto-starting never competes with a supervised watch.
  Until the model files are there, recall matches by exact words, and before anything is indexed it returns
  nothing; it never waits.

`--json` prints `{"schema": "rummage.project.v1", "cwd", "project"?, "history"?, "added", "started"}` as in the
hook's `rummage.project` above. `--no-start` only registers.

## Pausing and keeping things out

`rummage pause [recall|index|all] [--for 2h] [--here [DIR] | --session ID] [--private]` switches memory off for a
while; `rummage resume [recall|index|all] [--here [DIR] | --session ID]` switches it back (a pause with `--for`
also ends by itself). The reach of a resume must match the pause's.

- `recall`: the hook returns no leads; its reply carries `rummage.paused` and a `systemMessage` saying until when.
- `index`: nothing in reach is read meanwhile; whatever arrived is read on the first run after the pause ends.
- `--private`: nothing from that time, in that reach, is ever kept, not even after the pause. It records an
  exclusion mark.

`rummage exclude --since T --until T | --session ID | --here [DIR] | -s SOURCE` marks something after the fact: what
it reaches is dropped from the index now and never read again (`T` is `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM` or a duration
ago like `2h`). `rummage exclude --list` shows the marks; `rummage exclude --remove ID` removes one, and the next
index run reads its items again (it re-reads the sources from the start, skipping what is already kept). Marks only
change the index; rummage never changes the chats or files it reads. `rummage status` shows pauses and marks.

For slash commands, `pause --hook` and `resume --hook` read a command hook's JSON (Claude Code's
`UserPromptExpansion`: `command_args`, `cwd`, `session_id`) and reply `{"decision": "block", "reason": "<one line>"}`.
The words in `command_args` pick what and where: `recall`, `index` or `all`; a duration (`30m`, `2h`); `here` (the
hook's `cwd`); `session` (the hook's session); `private`.

## Claude Code and Codex

This repository is a plugin marketplace for both. The plugin holds the recall hook (a `UserPromptSubmit` hook running
`rummage recall --hook --harness claude-code`, or `--harness codex` from Codex's own manifest
`.codex-plugin/plugin.json`) and the skill. Installing it is the choice to have that harness's chats and the
projects you open remembered: the first prompt in a new project says once, under the prompt, what rummage now
reads and how to stop it.

In Claude Code, `/rummage:pause` and `/rummage:resume` take the words of `rummage pause --hook` (`recall`, `index`
or `all`, a time like `2h`, `here`, `session`, `private`); a `UserPromptExpansion` hook answers them directly, so
they cost no model turn. In Codex, run `rummage pause` and `rummage resume` in a terminal.

Claude Code:

```
/plugin marketplace add soffits/rummage
/plugin install rummage@rummage
```

Codex:

```bash
codex plugin marketplace add soffits/rummage
codex plugin add rummage@rummage
```

Codex runs a new hook only after you trust it: open `/hooks` once and trust the rummage hook.

Without the plugin, the same hook goes in `~/.claude/settings.json` or `~/.codex/hooks.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "rummage recall --hook --harness claude-code", "timeout": 5 } ] }
    ]
  }
}
```

### Indexing progress in Claude Code

While the index is still filling up, paused or broken, the recall hook shows one line under your prompt
("rummage: indexing 340/1,200, ~6 min left"), and the model gets the same news in a sentence. Nothing shows once it's
complete. To keep the line in view between prompts, add rummage to your status line in `~/.claude/settings.json`
(plugins can't set it for you):

```json
{ "statusLine": { "type": "command", "command": "rummage status --line", "refreshInterval": 5 } }
```

It prints nothing when the index is complete and nothing is paused, and reads two small files, so a short interval
costs little. A pause covering the folder comes first ("rummage: recall paused until 16:30"). On
Windows the same command works whether Claude Code runs it through Git Bash or PowerShell. If you already have a
status line script, print `rummage status --line` from it too.

## OpenCode

The plugin in [integrations/opencode](../integrations/opencode) targets the OpenCode 2 plugin API. It runs recall once
when a prompt is admitted and keeps the block in that message's metadata, then adds it to the message in each model
request, so the text you typed stays as you typed it. It also registers the rummage skill. Add the package to
`opencode.jsonc`:

```jsonc
{
  "plugins": [
    "@soffits/opencode-rummage",
    // or with options: { "package": "@soffits/opencode-rummage", "options": { "command": "rummage", "timeout": 5000, "args": [] } }
  ]
}
```

`args` are passed on to `rummage recall` (for example `["-s", "notes"]`); `"skill": false` leaves the skill out.

The package's TUI part shows indexing progress (and a pause) in the footer while the index is unfinished, paused or
broken, a toast when indexing starts, finishes or fails, and once a note on what the plugin does. `/rummage-pause`
and `/rummage-resume` take the same words as in Claude Code. Between index runs it only checks the progress file's time stamp.

## Hermes Agent

The plugin in [integrations/hermes](../integrations/hermes) adds a `pre_llm_call` hook; Hermes appends what it returns
to the turn's user message. `RUMMAGE_COMMAND` and `RUMMAGE_RECALL_TIMEOUT` (seconds) change the command and timeout.

```bash
hermes plugins install soffits/rummage/integrations/hermes --enable
hermes skills install soffits/rummage/skills/rummage
```

Hermes has no status area for plugins: the model still hears when the index is incomplete, and
`rummage status --watch` in another terminal (or `rummage status --line` in your prompt) shows the progress. The same
goes for Codex, whose hook output has no status line.

## DeepSeek Harness

A plugin that listens on `agent/pre-step` and appends recall's block as a source-tagged user message on the first
step of a turn, following the pattern of the shipped `dsh-time-context` plugin:

```js
import { execFile } from "node:child_process"
import { promisify } from "node:util"
import { createUserMessage } from "@deepseek-ai/dsh-llm"

const run = promisify(execFile)
export const name = "rummage-recall"

export function apply(ctx) {
  ctx.on("agent/pre-step", async ({ step }, next) => {
    const decision = await next()
    if (decision.kind === "reject" || step !== 1) return decision
    const last = [...decision.messages].reverse().find((m) => m.source?.kind === "user")
    const prompt = last?.content?.filter((p) => p.type === "text").map((p) => p.text).join("\n")
    if (!prompt) return decision
    const { stdout } = await run("rummage", ["recall", prompt], { timeout: 5000 }).catch(() => ({ stdout: "" }))
    const text = stdout.trim()
    if (!text) return decision
    return { ...decision, messages: [...decision.messages, createUserMessage({
      content: [{ type: "text", text }], source: { kind: name, form: "snapshot", sections: [{ name, text }] },
    })] }
  })
}
```

## Index progress for adapters

`rummage recall --hook` reads a JSON object on stdin: `prompt`, `session_id`, and `cwd`, `harness` and `context` (the
turn the prompt answers) when the harness knows them. It always replies with JSON, also when it has nothing to recall:

- `hookSpecificOutput.additionalContext`: the block for the model, when there is one. It ends with one sentence on
  the unfinished index only when that changes how to read it: a result's source is still being read, a tenth or more
  of the passages have no vectors yet, or the last run stopped or failed; then, when nothing matched, it is only that
  sentence.
- `systemMessage`: the status line for the person, only while the index isn't complete (or the pause, while recall
  is paused).
- `rummage.index`: the whole summary, the same object `rummage status --line --json` prints:

| Field | Meaning |
|---|---|
| `state` | `complete`, `building`, `paused`, `stopped` (a run died part way) or `error` |
| `line` | short status-bar text, `""` when complete |
| `note` | one sentence for a model, `""` when complete |
| `items` | `{done, total}` items read by a run that is reading now, else `null` |
| `vectors` | `{done, total}` passages with a vector of the model searches use, else `null` |
| `eta_seconds` | time left for those vectors at the recent pace, else `null` |
| `paused`, `error` | why work stopped with some left, what failed, else `null` |
| `updated` | when a run last wrote progress (Unix time), else `null` |
| `file` | the progress file; poll its modification time and run `status --line --json` only when it changes |

`status --line --json` adds `paused_memory`, the pause covering the current folder (or `null`), and puts its line
first in `line`. The hook reply adds `rummage.notice` on the prompt where `ensure` started remembering something
new, and shows it in `systemMessage`.

`search --json`, `get --json` and `recall --json` carry `state`, `note`, `items`, `vectors`, `eta_seconds`, `paused`
and `error` as `index` while the index is incomplete, and leave it out once it's complete.

## Any other agent

Call `rummage recall "<prompt>"` before the model sees a message and add its output, if any, as context; or give
the agent the skill and let it call `rummage search` itself when it needs to.
