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
  out (`--session`, or the hook's `session_id`). Credentials are masked in the block unless `--show-secrets`.
- **`rummage recent`** lists the latest chat turns and changed files across sources (`--minutes`, `-s`,
  `--exclude-session`, `--json`), and **`rummage get <#id> --around N`** shows a chat hit with the turns around it.

The adapters below call `rummage recall` before each prompt and add its block next to the user's message, never to
the system prompt, so the prompt cache keeps working. A recall that fails or takes longer than its timeout (5 s)
adds nothing and the turn goes on. Each one assumes `rummage` is on the agent's `PATH`.

## Claude Code and Codex

This repository is a plugin marketplace for both. The plugin holds the recall hook (`hooks/hooks.json`, a
`UserPromptSubmit` hook running `rummage recall --hook`) and the skill.

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
      { "hooks": [ { "type": "command", "command": "rummage recall --hook", "timeout": 5 } ] }
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

It prints nothing when the index is complete, and reads one small file, so a short interval costs little. On
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

The package's TUI part shows indexing progress in the footer while the index is unfinished, paused or broken, and a
toast when indexing starts, finishes or fails. Between index runs it only checks the progress file's time stamp.

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

`rummage recall --hook` reads a JSON object on stdin: `prompt`, `session_id`, and `cwd` and `harness` when the
harness knows them. It always replies with JSON, also when it has nothing to recall:

- `hookSpecificOutput.additionalContext`: the block for the model, when there is one. While the index is
  incomplete it ends with one sentence saying so; when nothing matched it is only that sentence.
- `systemMessage`: the status line for the person, only while the index isn't complete.
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

`search --json`, `get --json` and `recall --json` carry `state`, `note`, `items`, `vectors`, `eta_seconds`, `paused`
and `error` as `index` while the index is incomplete, and leave it out once it's complete.

## Any other agent

Call `rummage recall "<prompt>"` before the model sees a message and add its output, if any, as context; or give
the agent the skill and let it call `rummage search` itself when it needs to.
