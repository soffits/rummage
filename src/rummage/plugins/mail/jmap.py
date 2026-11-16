"""Reading a JMAP mail account (RFC 8620/8621) read-only: query message ids, fetch blob ids, download raw messages."""

from __future__ import annotations

import base64
import datetime as dt
import json
import urllib.parse
import urllib.request

from rummage import net
from rummage.sources import Plan

MAIL = "urn:ietf:params:jmap:mail"
CORE = "urn:ietf:params:jmap:core"
SKIP_ROLES = {"junk", "trash", "drafts"}
PAGE = 500


class Client:
    def __init__(self, session_url: str, user: str, password: str):
        self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        self.session = self._json(urllib.request.Request(session_url))
        self.api = self.session["apiUrl"]
        self.account = self.session["primaryAccounts"][MAIL]

    def _json(self, request: urllib.request.Request):
        request.add_header("Authorization", self.auth)
        with net.urlopen(request, timeout=120) as r:
            return json.load(r)

    def call(self, method: str, args: dict) -> dict:
        body = json.dumps({"using": [CORE, MAIL], "methodCalls": [[method, {"accountId": self.account, **args}, "0"]]})
        request = urllib.request.Request(self.api, data=body.encode(), headers={"Content-Type": "application/json"})
        name, result, _ = self._json(request)["methodResponses"][0]
        if name == "error":
            raise RuntimeError(f"JMAP {method}: {result.get('type')}: {result.get('description', '')}")
        return result

    def download(self, blob: str) -> bytes:
        url = self.session["downloadUrl"]
        for key, value in (("accountId", self.account), ("blobId", blob), ("type", "message/rfc822"),
                           ("name", "message.eml")):
            url = url.replace("{" + key + "}", urllib.parse.quote(value, safe=""))
        request = urllib.request.Request(url)
        request.add_header("Authorization", self.auth)
        with net.urlopen(request, timeout=120) as r:
            return r.read()


def plan_jmap(source, known: dict[str, str | None]) -> Plan:
    from . import message_item

    client = Client(source.spec["jmap_session"], source.spec["jmap_user"], source.password())
    boxes = client.call("Mailbox/get", {"ids": None, "properties": ["role", "name"]})["list"]
    skip = [b["id"] for b in boxes if (b.get("role") or "") in SKIP_ROLES]
    conditions: list[dict] = [{"inMailboxOtherThan": skip}] if skip else []
    day = source.since_date()
    if day:
        conditions.append({"after": f"{day.isoformat()}T00:00:00Z"})
    query = {"filter": {"operator": "AND", "conditions": conditions} if conditions else None,
             "sort": [{"property": "receivedAt", "isAscending": True}], "limit": PAGE}
    ids: list[str] = []
    while True:
        page = client.call("Email/query", {**query, "position": len(ids)})["ids"]
        ids += page
        if len(page) < PAGE:
            break
    prefix = "jmap/"
    current = {prefix + i for i in ids}
    new = [i for i in ids if known.get(prefix + i) is None]

    def upserts():
        for start in range(0, len(new), 50):
            emails = client.call("Email/get", {"ids": new[start:start + 50], "properties": ["blobId"]})["list"]
            for email in emails:
                raw = client.download(email["blobId"])
                yield message_item(source.name, prefix + email["id"],
                                   f"mail:{source.spec['jmap_user']}/{email['id']}", raw)

    return Plan(upserts(), [k for k in known if k not in current], {}, len(new))
