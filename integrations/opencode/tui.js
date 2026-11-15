// OpenCode TUI part of the rummage plugin: shows rummage's index state in the footer while it is building, paused
// or broken (or memory is paused), toasts when indexing starts, finishes or fails, a one-time note on what the plugin
// does, and /rummage-pause and /rummage-resume.
//
// All state comes from `rummage status --line --json`, which reads one small progress file. Between runs this part
// only checks that file's modification time, so an idle index costs a stat every few seconds. Written as plain JS
// (no JSX): OpenCode does not transpile JSX in packages installed under node_modules.

import { execFile } from "node:child_process"
import { stat } from "node:fs/promises"
import { createElement, effect, insert, setProp } from "@opentui/solid"
import { createSignal } from "solid-js"

const CHECK_MS = 2000
const RECHECK_BUSY_MS = 30000
const FOOTER_CHARS = 60
const KEY = Symbol.for("rummage.opencode.tui")

const INTRO =
  "rummage remembers the project you open and OpenCode's own chats, and adds passages that clearly match to your " +
  "prompts. /rummage-pause stops it for a while; removing the plugin turns it off."

function run(command, args, timeout, input) {
  return new Promise((resolve) => {
    const child = execFile(command, args, { timeout, maxBuffer: 1 << 20, windowsHide: true }, (error, stdout, stderr) =>
      resolve({ ok: !error, stdout: String(stdout ?? ""), stderr: String(stderr ?? "") }),
    )
    child.stdin.on("error", () => {})
    child.stdin.end(input ?? "")
  })
}

export default {
  id: "rummage.tui",
  setup(context) {
    // The TUI may set a plugin up again while an older instance lives; only the newest one polls and renders.
    globalThis[KEY]?.()
    const options = context.options ?? {}
    const command = options.command ?? "rummage"
    const [info, setInfo] = createSignal({ state: "complete", line: "" })
    const started = Date.now()
    let file, mtime = 0, checked = 0, timer, stopped = false

    async function refresh() {
      const result = await run(command, ["status", "--line", "--json"], 10000)
      checked = Date.now()
      if (!result.ok) return
      let next
      try {
        next = JSON.parse(result.stdout)
      } catch {
        return
      }
      const before = info()
      file = next.file ?? file
      setInfo(next)
      if (before.state !== next.state && checked > started + 1000) {
        if (next.state === "complete" && before.state === "building")
          context.ui.toast.show({ title: "rummage", message: "index complete", variant: "success" })
        else if (next.state === "error" || next.state === "stopped")
          context.ui.toast.show({ title: "rummage", message: next.note || next.line, variant: "warning" })
        else if (next.state === "building" && before.state === "complete")
          context.ui.toast.show({ title: "rummage", message: next.line, variant: "info" })
      }
    }

    async function tick() {
      if (stopped) return
      try {
        const now = Date.now()
        let changed = false
        if (file) {
          const seen = await stat(file).then((s) => s.mtimeMs, () => 0)
          changed = seen !== mtime
          mtime = seen
        }
        const busy = info().state !== "complete" || info().line
        if (!file || changed || (busy && now - checked > RECHECK_BUSY_MS)) await refresh()
      } finally {
        if (!stopped) timer = setTimeout(tick, CHECK_MS)
      }
    }

    void tick()

    function label() {
      const el = createElement("text")
      insert(el, () => {
        const line = info().line || ""
        return line.length > FOOTER_CHARS ? line.slice(0, FOOTER_CHARS - 3) + "..." : line
      })
      effect((prev) => {
        const bad = ["error", "stopped", "paused"].includes(info().state) || Boolean(info().paused_memory)
        const theme = context.theme.text
        return setProp(el, "fg", bad ? (theme.feedback?.warning?.base ?? theme.base) : theme.muted, prev)
      })
      return el
    }

    const unslot = [
      context.ui.slot({ append: "prompt.footer.status", render: label }),
      context.ui.slot({ append: "home.footer.status", render: label }),
    ]

    const directory = () => (context.location ?? context.data.location.default())?.directory
    const sessionID = () => {
      const route = context.ui.router.current()
      return route.type === "session" ? route.sessionID : undefined
    }

    // The same words as Claude Code's /rummage:pause: recall|index|all, a time like 2h, here, session, private.
    async function pauseCommand(verb, input) {
      const event = { command_args: String(input ?? ""), cwd: directory(), session_id: sessionID() }
      const result = await run(command, [verb, "--hook"], 10000, JSON.stringify(event))
      let reason = ""
      try {
        reason = JSON.parse(result.stdout).reason ?? ""
      } catch {}
      context.ui.toast.show({
        title: "rummage",
        message: reason || result.stderr.trim() || `${verb} failed`,
        variant: result.ok ? "info" : "error",
      })
      void refresh()
    }

    const unkeymap = context.keymap.layer(() => ({
      mode: "global",
      commands: [
        {
          id: "rummage.pause",
          title: "Pause rummage (recall, index or all; a time like 2h; here or session)",
          group: "rummage",
          palette: true,
          slash: { name: "rummage-pause", arguments: true },
          run: (input) => pauseCommand("pause", input),
        },
        {
          id: "rummage.resume",
          title: "Resume rummage",
          group: "rummage",
          palette: true,
          slash: { name: "rummage-resume", arguments: true },
          run: (input) => pauseCommand("resume", input),
        },
      ],
    }))

    const [seen, setSeen] = context.storage.store("rummage.intro", { initial: { shown: false } })
    if (!seen.shown) {
      context.ui.toast.show({ title: "rummage", message: INTRO, variant: "info" })
      void setSeen((draft) => {
        draft.shown = true
      })
    }

    const dispose = () => {
      if (stopped) return
      stopped = true
      clearTimeout(timer)
      for (const off of unslot) off?.()
      unkeymap?.()
      if (globalThis[KEY] === dispose) delete globalThis[KEY]
    }
    globalThis[KEY] = dispose
    return dispose
  },
}
