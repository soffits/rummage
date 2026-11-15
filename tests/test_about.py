import json
import time

from rummage.cli import main
from rummage.store import Item, Segment, Store


def test_about_gathers_names_and_builds_a_timeline(tmp_path, capsys):
    store = Store(tmp_path / "data" / "index.db")
    now = time.time()
    rows = [
        ("notes", "people/bob.md", "note", "Bob Smith", now - 9000, {"aliases": ["Bob", "老张"]},
         "Bob Smith is Alice's partner."),
        ("notes", "trip.md", "note", "Trip", now - 5000, {}, "老张 booked the train tickets to Montreal."),
        ("chats", "s1:7", "chat", "Bob, chat", now - 100, {"speaker": "Bob", "session": "s1"},
         "Bob: can you check the pencil order?"),
        ("notes", "other.md", "note", "Other", now - 50, {}, "Nothing about anyone here."),
    ]
    for source, key, kind, title, mtime, info, text in rows:
        store.put(Item(source, key, kind, title, key, mtime, None, info), [Segment("text", text, loc="0")])
    store.commit()
    store.close()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[people.'Bob Smith']\naliases = ['Bobby']\n", encoding="utf-8")
    assert main(["--config", str(cfg), "about", "Bob Smith", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc["names"]) >= {"Bob Smith", "Bob", "老张", "Bobby"}
    assert [r["ref"] for r in doc["results"]] == ["s1:7", "trip.md", "people/bob.md"]
    assert doc["results"][0]["how"] == "wrote"
