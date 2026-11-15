// OpenCode plugin for rummage: recall from the user's files on each prompt, plus the rummage skill.
//
// The prompt hook runs `rummage recall` once per prompt and keeps the block in the message's metadata, so it is
// stored with the message and never re-run. The context hook adds that block to its user message in every model
// request, so the system prompt stays the same and the request prefix stays cacheable. A recall that fails or
// takes longer than `timeout` adds nothing.

import { execFile } from "node:child_process"
import { readFileSync } from "node:fs"

const KEY = "rummage.recall"

function recall(command, args, prompt, sessionID, timeout) {
  return new Promise((resolve) => {
    const child = execFile(
      command,
      ["recall", "--hook", ...args],
      { timeout, maxBuffer: 1 << 20, windowsHide: true },
      (error, stdout) => {
        if (error || !stdout.trim()) return resolve("")
        try {
          resolve(JSON.parse(stdout).hookSpecificOutput?.additionalContext ?? "")
        } catch {
          resolve("")
        }
      },
    )
    child.stdin.on("error", () => {})
    child.stdin.end(JSON.stringify({ prompt, session_id: sessionID }))
  })
}

function bundledSkill() {
  try {
    const path = new URL("./skills/rummage/SKILL.md", import.meta.url)
    const text = readFileSync(path, "utf8")
    const [, front = "", body = text] = text.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/) ?? []
    const field = (name) => front.match(new RegExp(`^${name}:\\s*(.*)$`, "m"))?.[1].trim()
    return { id: field("name") ?? "rummage", description: field("description"), path: path.pathname, content: body }
  } catch {
    return undefined
  }
}

export default {
  id: "rummage",
  async setup(ctx) {
    const options = ctx.options ?? {}
    const command = options.command ?? "rummage"
    const args = options.args ?? []
    const timeout = options.timeout ?? 5000

    await ctx.session.hook("prompt", async (event) => {
      const text = event.prompt.text ?? ""
      if (!text.trim()) return
      const block = await recall(command, args, text, String(event.sessionID), timeout)
      if (block) event.metadata = { ...event.metadata, [KEY]: block }
    })

    await ctx.session.hook("context", (event) => {
      event.messages.forEach((message, index) => {
        const block = message.role === "user" && message.metadata?.[KEY]
        if (typeof block !== "string") return
        event.messages[index] = Object.assign(Object.create(Object.getPrototypeOf(message)), message, {
          content: [...message.content, { type: "text", text: block }],
        })
      })
    })

    const skill = options.skill === false ? undefined : bundledSkill()
    if (skill) {
      await ctx.skill.transform((editor) => {
        if (!editor.get(skill.id)) editor.add({ ...skill, name: skill.id })
      })
    }
  },
}
