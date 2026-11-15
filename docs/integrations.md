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

## Hermes Agent

The plugin in [integrations/hermes](../integrations/hermes) adds a `pre_llm_call` hook; Hermes appends what it returns
to the turn's user message. `RUMMAGE_COMMAND` and `RUMMAGE_RECALL_TIMEOUT` (seconds) change the command and timeout.

```bash
hermes plugins install soffits/rummage/integrations/hermes --enable
hermes skills install soffits/rummage/skills/rummage
```

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

## Any other agent

Call `rummage recall "<prompt>"` before the model sees a message and add its output, if any, as context; or give
the agent the skill and let it call `rummage search` itself when it needs to.
