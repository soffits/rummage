"""Mail, read-only: a local Maildir or mbox, an IMAP account, or a JMAP account. One item per message: subject,
sender, recipients, date, the body (plain text, else the HTML's text) and the text of attached documents.

    [sources.mail]
    type = "mail"
    maildir = "~/Mail/personal"          # or: mbox = "~/mail/archive.mbox"
    # IMAP instead:
    # imap_host = "imap.example.com"
    # imap_user = "me@example.com"
    # JMAP instead:
    # jmap_session = "https://mail.example.com/jmap/session"
    # jmap_user = "me@example.com"
    # password_env = "MAIL_PASSWORD"     # or password_file = "...", or password_command = ["pass", "mail"]
    # oauth_token_command = ["ortie", "token", "show", "-a", "work"]   # IMAP with an OAuth access token (XOAUTH2)
    # folders = "all"                    # IMAP: every folder but spam, trash and drafts (default); or a list
    # since = "2y"                       # only mail from then on: a date (2024-01-01) or a span (90d, 2y)

Nothing is changed on the server: IMAP folders are opened with EXAMINE and messages fetched with BODY.PEEK, so
nothing is marked read; JMAP is only queried and downloaded from. With `folders = "all"`, an IMAP account with an
"All Mail" folder (special-use \\All) is read from that folder alone, since it holds everything else; otherwise
every folder is read except those marked \\Junk, \\Trash or \\Drafts (or named like them). JMAP reads every
mailbox except those with the junk, trash and drafts roles. An IMAP folder's cursor is its UIDVALIDITY and
highest UID; a changed UIDVALIDITY reads it again. Messages that vanished are dropped.
"""

from __future__ import annotations

import datetime as dt
import email
import email.policy
import fnmatch
import hashlib
import imaplib
import mailbox
import os
import re
import shlex
import subprocess
from email.message import EmailMessage
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from rummage import extract as X
from rummage import text as T
from rummage.runtime import document_text
from rummage.sources import Plan, Source
from rummage.store import Item, Segment

ATTACHMENT_LIMIT = 32 << 20
# Folder names to leave out on servers that don't mark special-use folders.
SKIP_FOLDER_NAMES = ("*spam*", "*junk*", "*trash*", "*deleted*", "*drafts*", "*bin")
_LIST_LINE = re.compile(r'^\((?P<flags>[^)]*)\)\s+(?:"[^"]*"|NIL)\s+(?P<name>.+)$')


def _date(message: EmailMessage) -> float:
    try:
        return parsedate_to_datetime(message["Date"]).timestamp()
    except (TypeError, ValueError, IndexError):
        return 0.0


def _addresses(message: EmailMessage, field: str) -> str:
    return ", ".join(f"{name} <{addr}>" if name else addr for name, addr in getaddresses(message.get_all(field, [])))


def body_and_attachments(message: EmailMessage) -> tuple[str, list[tuple[str, str]]]:
    """The message's text, and `(file name, text)` for each attached document rummage can read."""
    part = message.get_body(preferencelist=("plain", "html"))
    body = ""
    if part is not None:
        content = part.get_content()
        body = X.html_text(content.encode("utf-8")) if part.get_content_subtype() == "html" else content
    attached = []
    for att in message.iter_attachments():
        name = att.get_filename() or ""
        data = att.get_payload(decode=True) or b""
        if not name or len(data) > ATTACHMENT_LIMIT:
            continue
        try:
            text = X.bytes_text(name, data)
        except Exception:
            text = None
        if text and text.strip():
            attached.append((name, text))
    return body.strip(), attached


def message_item(source: str, key: str, ref: str, raw: bytes, folder: str = "") -> Item:
    message = email.message_from_bytes(raw, policy=email.policy.default)
    subject = str(message["Subject"] or "(no subject)")
    when = _date(message)
    info = {"from": _addresses(message, "From"), "to": _addresses(message, "To"),
            "date": dt.datetime.fromtimestamp(when, dt.timezone.utc).isoformat(timespec="seconds") if when else "",
            "message_id": str(message["Message-ID"] or "")}
    if folder:
        info["folder"] = folder
    item = Item(source, key, "mail", subject, ref, when, hashlib.sha256(raw).hexdigest(), info)
    item._message = message  # type: ignore[attr-defined]
    return item


def message_segments(item: Item) -> list[Segment]:
    message = item._message  # type: ignore[attr-defined]
    body, attached = body_and_attachments(message)
    head = f"From: {item.info['from']}\nTo: {item.info['to']}\nSubject: {item.title}\n\n"
    segments = [Segment("name", " ".join(filter(None, [item.title, item.info["from"], item.info["to"]])))]
    for start, piece in T.chunks((head + body)[:X.MAX_CHARS]):
        seg = Segment("text", piece, loc=str(start))
        if T.has_prose(piece):
            seg.embed_text = document_text(item.title, piece)
        segments.append(seg)
    for name, text in attached:
        item.info.setdefault("attachments", []).append(name)
        for start, piece in T.chunks(text[:X.MAX_CHARS]):
            seg = Segment("text", piece, loc=f"{name}@{start}")
            if T.has_prose(piece):
                seg.embed_text = document_text(f"{item.title} / {name}", piece)
            segments.append(seg)
    return segments


def _logout(imap) -> None:
    try:
        imap.logout()
    except Exception:
        pass


class MailSource(Source):
    def __init__(self, name: str, spec: dict, data_dir: Path | None = None):
        self.name = name
        self.spec = spec
        self.since = spec.get("since")

    def watch_roots(self) -> list[Path]:
        for key in ("maildir", "mbox"):
            if self.spec.get(key):
                path = Path(os.path.expanduser(self.spec[key]))
                return [path if path.is_dir() else path.parent]
        return []

    def since_date(self) -> dt.date | None:
        """`since` as a day: a date, or a span back from today (`90d`, `2y`)."""
        value = str(self.since or "").strip()
        if not value:
            return None
        if value[-1] in "dy" and value[:-1].isdigit():
            days = int(value[:-1]) * (365 if value[-1] == "y" else 1)
            return dt.date.today() - dt.timedelta(days=days)
        return dt.date.fromisoformat(value)

    def _recent_enough(self, item: Item) -> bool:
        day = self.since_date()
        if not day or not item.mtime:
            return True
        return item.mtime >= dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp()

    def plan(self, known: dict[str, str | None], state: dict) -> Plan:
        if self.spec.get("imap_host"):
            return self._plan_imap(known, state)
        if self.spec.get("jmap_session"):
            from .jmap import plan_jmap
            return plan_jmap(self, known)
        local = self._local()
        upserts, current = [], set()
        for key, raw in local:
            current.add(key)
            if key in known:
                continue  # a stored message doesn't change
            item = message_item(self.name, key, f"mail:{key}", raw)
            if self._recent_enough(item):
                upserts.append(item)
        return Plan(upserts, [k for k in known if k not in current], {}, len(upserts))

    def _local(self):
        if self.spec.get("maildir"):
            box = mailbox.Maildir(os.path.expanduser(self.spec["maildir"]), factory=None, create=False)
        else:
            box = mailbox.mbox(os.path.expanduser(self.spec["mbox"]), factory=None, create=False)
        for key in box.keys():
            raw = box.get_bytes(key)
            # Maildir keys survive flag changes; mbox keys are positions, so name a message by its bytes there.
            yield (str(key) if self.spec.get("maildir") else hashlib.sha1(raw).hexdigest()[:20]), raw

    def _run(self, command) -> str:
        command = [os.path.expanduser(c) for c in command] if isinstance(command, list) else shlex.split(command)
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()

    def login(self, imap) -> None:
        user = self.spec["imap_user"]
        if self.spec.get("oauth_token_command"):
            token = self._run(self.spec["oauth_token_command"])
            imap.authenticate("XOAUTH2", lambda _: f"user={user}\x01auth=Bearer {token}\x01\x01".encode())
        else:
            imap.login(user, self.password())

    def password(self) -> str:
        if self.spec.get("password_env"):
            return os.environ[self.spec["password_env"]]
        if self.spec.get("password_command"):
            return self._run(self.spec["password_command"])
        return Path(os.path.expanduser(self.spec["password_file"])).read_text(encoding="utf-8").strip()

    def _folders(self, imap) -> list[str]:
        wanted = self.spec.get("folders", "all")
        if wanted != "all":
            return list(wanted)
        listed = []
        for line in imap.list()[1] or []:
            m = _LIST_LINE.match(line.decode("utf-8", "replace") if isinstance(line, bytes) else str(line))
            if m:
                listed.append((m["flags"].lower(), m["name"].strip('"')))
        everything = [name for flags, name in listed if "\\all" in flags]
        if everything:
            return everything[:1]
        skip_flags = ("\\junk", "\\trash", "\\drafts", "\\noselect", "\\nonexistent")
        return [name for flags, name in listed if not any(f in flags for f in skip_flags)
                and not any(fnmatch.fnmatch(name.lower(), pat) for pat in SKIP_FOLDER_NAMES)]

    def _connect(self):
        imap = imaplib.IMAP4_SSL(self.spec["imap_host"], int(self.spec.get("imap_port", 993)))
        self.login(imap)
        return imap

    def _plan_imap(self, known: dict[str, str | None], state: dict) -> Plan:
        """Lists what is new first (UIDs only), then fetches messages one at a time as the indexer asks for them,
        so a large mailbox never sits in memory."""
        imap = self._connect()
        removed, new_state, todo = [], {}, []
        try:
            for folder in self._folders(imap):
                typ, _ = imap.select(f'"{folder}"', readonly=True)
                if typ != "OK":
                    continue
                validity = imap.response("UIDVALIDITY")[1][0].decode()
                old = state.get(folder, {})
                start = old.get("uid", 0) + 1 if old.get("validity") == validity else 1
                criteria = ["UID", f"{start}:*"]
                if self.since_date():
                    criteria += ["SINCE", self.since_date().strftime("%d-%b-%Y")]
                typ, data = imap.uid("SEARCH", *criteria)
                uids = [int(u) for u in (data[0] or b"").split() if int(u) >= start]
                prefix = f"{folder}/{validity}/"
                if start == 1:
                    typ, data = imap.uid("SEARCH", "ALL")
                    present = {f"{prefix}{int(u)}" for u in (data[0] or b"").split()}
                    removed += [k for k in known if k.startswith(f"{folder}/") and k not in present]
                todo += [(folder, uid, f"{prefix}{uid}") for uid in uids if f"{prefix}{uid}" not in known]
                new_state[folder] = {"validity": validity, "uid": max(uids + [old.get("uid", 0) if start > 1 else 0])}
        finally:
            _logout(imap)

        def upserts():
            connection, selected = None, None
            try:
                for folder, uid, key in todo:
                    if connection is None:
                        connection = self._connect()
                    if folder != selected:
                        connection.select(f'"{folder}"', readonly=True)
                        selected = folder
                    typ, data = connection.uid("FETCH", str(uid), "(BODY.PEEK[])")
                    raw = next((d[1] for d in data if isinstance(d, tuple)), None)
                    if raw:
                        yield message_item(self.name, key, f"mail:{self.spec['imap_user']}/{key}", raw, folder)
            finally:
                if connection is not None:
                    _logout(connection)

        return Plan(upserts(), removed, new_state, len(todo))

    def load(self, item: Item) -> list[Segment]:
        return message_segments(item)
