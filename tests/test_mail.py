import mailbox
from email.message import EmailMessage

from rummage.plugins import mail as M


def _message(subject, body, html=False, attachment=None, date="Fri, 14 Nov 2026 10:00:00 -0500"):
    m = EmailMessage()
    m["From"] = "Clinic <desk@clinic.example>"
    m["To"] = "me@example.com"
    m["Subject"] = subject
    m["Date"] = date
    m["Message-ID"] = f"<{abs(hash(subject))}@clinic.example>"
    if html:
        m.set_content(f"<html><body><p>{body}</p><script>track()</script></body></html>", subtype="html")
    else:
        m.set_content(body)
    if attachment:
        m.add_attachment(attachment[1].encode(), maintype="text", subtype="plain", filename=attachment[0])
    return m


def test_maildir_messages_with_attachment_text(tmp_path):
    box = mailbox.Maildir(tmp_path / "Mail", create=True)
    box.add(_message("Appointment confirmed", "Your cleaning is on Tuesday at 3 pm.",
                     attachment=("prep.txt", "Please arrive ten minutes early and bring your health card.")))
    box.add(_message("Newsletter", "Our clinic hours change in December.", html=True))
    src = M.MailSource("mail", {"maildir": str(tmp_path / "Mail")})
    plan = src.plan({}, {})
    items = {i.title: i for i in plan.upserts}
    assert set(items) == {"Appointment confirmed", "Newsletter"}
    segs = src.load(items["Appointment confirmed"])
    text = " ".join(s.text for s in segs)
    assert "Tuesday at 3 pm" in text and "health card" in text and "desk@clinic.example" in text
    assert items["Appointment confirmed"].info["attachments"] == ["prep.txt"]
    assert "hours change in December" in " ".join(s.text for s in src.load(items["Newsletter"]))
    assert "<p>" not in " ".join(s.text for s in src.load(items["Newsletter"]))

    known = {i.key: i.fingerprint for i in plan.upserts}
    first_key = next(k for k, i in ((i.key, i) for i in plan.upserts) if i.title == "Newsletter")
    box.remove(first_key)
    again = src.plan(known, {})
    assert list(again.upserts) == [] and again.removed == [first_key]


def test_mbox_and_since(tmp_path):
    box = mailbox.mbox(tmp_path / "archive.mbox", create=True)
    box.add(_message("Old receipt", "Order 1", date="Mon, 01 Jan 2024 10:00:00 +0000"))
    box.add(_message("New receipt", "Order 2"))
    box.flush()
    src = M.MailSource("mail", {"mbox": str(tmp_path / "archive.mbox"), "since": "2025-01-01"})
    assert [i.title for i in src.plan({}, {}).upserts] == ["New receipt"]


class FakeIMAP:
    """imaplib.IMAP4_SSL stand-in: one folder, three messages, records whether anything could set \\Seen."""

    def __init__(self, host, port):
        self.messages = {1: _message("First", "one").as_bytes(), 2: _message("Second", "two").as_bytes(),
                         3: _message("Third", "three").as_bytes()}
        self.readonly = None
        self.fetches = []

    def login(self, user, password):
        assert password == "pw"

    def list(self):
        return "OK", [b'(\\HasNoChildren) "/" "INBOX"', b'(\\Junk) "/" "Junk"']

    def select(self, folder, readonly=False):
        self.readonly = readonly
        return "OK", [b"3"]

    def response(self, code):
        return code, [b"777"]

    def uid(self, command, *args):
        if command == "SEARCH":
            if args[0] == "ALL":
                return "OK", [" ".join(map(str, self.messages)).encode()]
            start = int(args[1].split(":")[0])
            return "OK", [" ".join(str(u) for u in self.messages if u >= start).encode()]
        self.fetches.append(args[1])
        return "OK", [(b"x", self.messages[int(args[0])]), b")"]

    def logout(self):
        pass


def test_imap_reads_read_only_and_resumes_by_uid(monkeypatch):
    fake = {}
    monkeypatch.setattr(M.imaplib, "IMAP4_SSL", lambda h, p: fake.setdefault("c", FakeIMAP(h, p)))
    monkeypatch.setenv("PW", "pw")
    src = M.MailSource("mail", {"imap_host": "imap.example", "imap_user": "me", "password_env": "PW"})
    plan = src.plan({}, {})
    assert [i.title for i in plan.upserts] == ["First", "Second", "Third"]
    assert fake["c"].readonly is True and set(fake["c"].fetches) == {"(BODY.PEEK[])"}
    assert plan.state == {"INBOX": {"validity": "777", "uid": 3}}
    fake.clear()
    known = {i.key: i.fingerprint for i in plan.upserts}
    again = src.plan(known, plan.state)
    assert list(again.upserts) == [] and again.removed == []
    # a message another reader version read is fetched again, by its UID, though the cursor is past it
    known["INBOX/777/2"] = None
    fake.clear()
    assert [i.title for i in src.plan(known, plan.state).upserts] == ["Second"]


def test_tracking_links_and_tokens_are_not_read_as_words(tmp_path):
    token = "J8QQpnPzeSK3KNfk7OTgzF6IP6FqOryJAvdBbvwwUCnCDHiMh8g3XI6vHtDwcnwmH4bNEKMb"
    body = ("Weekly letter about community gardens.\u034f \u00ad\u034f \u00ad\u034f\n"
            "Trackers removed. More <https://relay.example/email/report#" + token + "-iCJjZIgIuTEFSqSB>\n"
            "Reply-To: <reply+8kq2x&71vbza&&482374f6c43e45ef0bd045e56e219423aeeb76c0e5ed8ebd77a602c1_at_mg1.example>\n"
            "Read it at https://letters.example/p/gardens and order 4815162342.")
    box = mailbox.Maildir(tmp_path / "Mail", create=True)
    box.add(_message("Gardens", body))
    src = M.MailSource("mail", {"maildir": str(tmp_path / "Mail")})
    text = " ".join(s.text for s in src.load(next(iter(src.plan({}, {}).upserts))))
    assert "community gardens" in text and "https://relay.example/…" in text
    assert "https://letters.example/p/gardens" in text and "4815162342" in text
    assert token[:20] not in text and "482374f6c43e" not in text and "\u034f" not in text and "\u00ad" not in text
    assert src.reader_of("any") == M.READER


def test_imap_reads_all_mail_alone_or_every_folder_but_junk_and_trash(monkeypatch, tmp_path):
    lists = {
        "gmail": [b'(\\HasNoChildren) "/" "INBOX"', b'(\\All \\HasNoChildren) "/" "[Gmail]/All Mail"',
                  b'(\\Junk) "/" "[Gmail]/Spam"', b'(\\Trash) "/" "[Gmail]/Trash"'],
        "plain": [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Archive"', b'() "/" "Spam"',
                  b'(\\Trash) "/" "Deleted Messages"', b'(\\Drafts) "/" "Drafts"', b'(\\Noselect) "/" "[Gmail]"'],
    }
    src = M.MailSource("mail", {"imap_host": "x", "imap_user": "me"})

    class Lister:
        def __init__(self, lines):
            self.lines = lines

        def list(self):
            return "OK", self.lines

    assert src._folders(Lister(lists["gmail"])) == ["[Gmail]/All Mail"]
    assert src._folders(Lister(lists["plain"])) == ["INBOX", "Archive"]

    script = tmp_path / "pw.py"
    script.write_text("print('from-command')", encoding="utf-8")
    import sys
    with_command = M.MailSource("mail", {"imap_host": "x", "imap_user": "me",
                                         "password_command": [sys.executable, str(script)]})
    assert with_command.password() == "from-command"
    assert M.MailSource("m", {"maildir": "x", "since": "30d"}).since_date() is not None


def test_jmap_account_read_only(tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    raw = {"e1": _message("Flight booked", "AC 123 departs at 9:40").as_bytes(),
           "e2": _message("Spam offer", "buy now").as_bytes()}
    seen_methods = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, binary=False):
            data = body if binary else json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            assert self.headers["Authorization"].startswith("Basic ")
            assert self.headers["User-Agent"].startswith("rummage/")   # servers that refuse Python's default
            if self.path == "/session":
                port = self.server.server_port
                return self._send({"apiUrl": f"http://127.0.0.1:{port}/api",
                                   "downloadUrl": f"http://127.0.0.1:{port}/dl/{{accountId}}/{{blobId}}/{{name}}",
                                   "primaryAccounts": {"urn:ietf:params:jmap:mail": "A1"}})
            return self._send(raw[self.path.split("/")[3].replace("blob-", "")], binary=True)

        def do_POST(self):
            call = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["methodCalls"][0]
            method, args = call[0], call[1]
            seen_methods.append(method)
            if method == "Mailbox/get":
                result = {"list": [{"id": "inbox", "role": "inbox"}, {"id": "junk", "role": "junk"}]}
            elif method == "Email/query":
                skipped = args["filter"]["conditions"][0]["inMailboxOtherThan"]
                result = {"ids": [] if args["position"] else (["e1"] if skipped == ["junk"] else ["e1", "e2"])}
            else:
                result = {"list": [{"id": i, "blobId": f"blob-{i}"} for i in args["ids"]]}
            self._send({"methodResponses": [[method, result, "0"]]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        src = M.MailSource("mail", {"jmap_session": f"http://127.0.0.1:{server.server_port}/session",
                                    "jmap_user": "me@example.com", "password_env": "PW", "since": "2y"})
        import os
        os.environ["PW"] = "pw"
        items = list(src.plan({}, {}).upserts)
        assert [i.title for i in items] == ["Flight booked"]
        assert "9:40" in " ".join(s.text for s in src.load(items[0]))
        assert set(seen_methods) <= {"Mailbox/get", "Email/query", "Email/get"}   # nothing that changes mail
    finally:
        server.shutdown()


def test_imap_xoauth2_sends_the_token(tmp_path):
    import sys
    script = tmp_path / "token.py"
    script.write_text("print('ya29.token')", encoding="utf-8")
    src = M.MailSource("school", {"imap_host": "x", "imap_user": "me@school.example",
                                  "oauth_token_command": [sys.executable, str(script)]})
    sent = {}

    class Auth:
        def authenticate(self, mechanism, callback):
            sent["mechanism"], sent["payload"] = mechanism, callback(b"")

        def login(self, *a):
            raise AssertionError("no password login with a token")

    src.login(Auth())
    assert sent == {"mechanism": "XOAUTH2", "payload": b"user=me@school.example\x01auth=Bearer ya29.token\x01\x01"}


def test_copies_of_one_mail_in_two_folders_are_one_hit(tmp_path):
    from rummage.search import Searcher
    from rummage.store import Item, Segment, Store

    store = Store(tmp_path / "index.db")
    for key, folder, mtime in (("inbox:7", "INBOX", 100.0), ("archive:3", "Archive", 5_000_000.0)):
        info = {"message_id": "<abc@shop.example>", "folder": folder}
        store.put(Item("mail", key, "mail", "Your order shipped", f"mail:{key}", mtime, key, info),
                  [Segment("text", "Your order with tracking number 1Z999 has shipped.", loc="0")])
    store.put(Item("mail", "inbox:8", "mail", "Other", "mail:inbox:8", 100.0, "x", {"message_id": "<other@x>"}),
              [Segment("text", "Another order has shipped too.", loc="0")])
    store.commit()
    hits = Searcher(store).search("order shipped", None)
    assert len(hits) == 2
    copy = next(h for h in hits if "1Z999" in h.snippet)
    assert {copy.ref, *copy.also} == {"mail:inbox:7", "mail:archive:3"}
    assert store.same_message(store.find("mail:inbox:7")) == ["mail:archive:3"]
