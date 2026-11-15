"""`rummage service install|uninstall|show`: run `rummage watch` under the operating system's own supervisor, so it
starts at login and comes back if it stops: a systemd user unit on Linux, a launchd agent on macOS, a scheduled
task at logon on Windows. `show` prints what `install` would write, without touching anything.
"""

from __future__ import annotations

import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

NAME = "rummage-watch"
LABEL = "io.github.soffits.rummage.watch"


def command(config: Path) -> list[str]:
    return [sys.executable, "-m", "rummage", "--config", str(config), "watch"]


def systemd_unit(config: Path) -> str:
    return f"""[Unit]
Description=rummage: keep the search index current

[Service]
ExecStart={shlex.join(command(config))}
Restart=on-failure
RestartSec=30
Nice=10

[Install]
WantedBy=default.target
"""


def launchd_plist(config: Path, log: Path) -> str:
    args = "\n".join(f"    <string>{escape(a)}</string>" for a in command(config))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>Nice</key><integer>10</integer>
  <key>StandardErrorPath</key><string>{escape(str(log))}</string>
</dict>
</plist>
"""


def windows_task(config: Path) -> list[str]:
    run = subprocess.list2cmdline(command(config))
    return ["schtasks", "/Create", "/TN", NAME, "/SC", "ONLOGON", "/RL", "LIMITED", "/F", "/TR", run]


def _paths(log_dir: Path) -> dict:
    system = platform.system()
    if system == "Linux":
        return {"kind": "systemd", "file": Path.home() / ".config" / "systemd" / "user" / f"{NAME}.service"}
    if system == "Darwin":
        return {"kind": "launchd", "file": Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist",
                "log": log_dir / "watch.log"}
    if system == "Windows":
        return {"kind": "schtasks"}
    raise SystemExit(f"no service support for {system}: run `rummage watch` under your own supervisor")


def show(config: Path, log_dir: Path) -> str:
    where = _paths(log_dir)
    if where["kind"] == "systemd":
        return f"# {where['file']}\n{systemd_unit(config)}"
    if where["kind"] == "launchd":
        return f"<!-- {where['file']} -->\n{launchd_plist(config, where['log'])}"
    return subprocess.list2cmdline(windows_task(config))


def install(config: Path, log_dir: Path) -> None:
    where = _paths(log_dir)
    if where["kind"] == "systemd":
        where["file"].parent.mkdir(parents=True, exist_ok=True)
        where["file"].write_text(systemd_unit(config), encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", f"{NAME}.service"], check=True)
    elif where["kind"] == "launchd":
        log_dir.mkdir(parents=True, exist_ok=True)
        where["file"].parent.mkdir(parents=True, exist_ok=True)
        where["file"].write_text(launchd_plist(config, where["log"]), encoding="utf-8")
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(where["file"])], check=True)
    else:
        subprocess.run(windows_task(config), check=True)
        subprocess.run(["schtasks", "/Run", "/TN", NAME], check=True)


def uninstall(log_dir: Path) -> None:
    where = _paths(log_dir)
    if where["kind"] == "systemd":
        subprocess.run(["systemctl", "--user", "disable", "--now", f"{NAME}.service"])
        where["file"].unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"])
    elif where["kind"] == "launchd":
        subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"])
        where["file"].unlink(missing_ok=True)
    else:
        subprocess.run(["schtasks", "/End", "/TN", NAME])
        subprocess.run(["schtasks", "/Delete", "/TN", NAME, "/F"])
