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

The snippets below call `rummage recall` from each tool's prompt hook. Adjust the command path if rummage is not on
the hook's `PATH`. `recall --hook` follows the documented `UserPromptSubmit` input and output of Claude Code and
Codex; the OpenCode, Hermes Agent and DeepSeek Harness plugins are short starting points written against those
tools' plugin docs and have not been run end to end yet.

## Claude Code

`~/.claude/settings.json` (or the project's `.claude/settings.json`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "rummage recall --hook", "timeout": 5 } ] }
    ]
  }
}
```

`--hook` reads the hook's JSON from stdin and replies with `hookSpecificOutput.additionalContext`.

## Codex

`~/.codex/hooks.json` (or `[hooks]` in `config.toml`):

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

A plugin file, `.opencode/plugin/rummage.ts` (or `~/.config/opencode/plugin/rummage.ts`):

```ts
import type { Plugin } from "@opencode-ai/plugin"

export const Rummage: Plugin = async ({ $ }) => ({
  "chat.message": async (_input, output) => {
    const prompt = output.parts.filter((p) => p.type === "text").map((p) => p.text).join("\n")
    const context = (await $`rummage recall ${prompt}`.nothrow().quiet()).stdout.toString().trim()
    if (context) output.parts.push({ type: "text", text: context, synthetic: true } as any)
  },
})
```

## Hermes Agent

A plugin whose `register` adds a `pre_llm_call` hook; Hermes joins the returned context into the user message:

```python
import subprocess


def recall(user_message: str = "", **_):
    run = subprocess.run(["rummage", "recall", user_message], capture_output=True, text=True, timeout=5)
    return {"context": run.stdout.strip()} if run.stdout.strip() else None


def register(ctx):
    ctx.register_hook("pre_llm_call", recall)
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
