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
    again = src.plan({i.key: i.fingerprint for i in plan.upserts}, plan.state)
    assert list(again.upserts) == [] and again.removed == []
