import json
import time

from rummage.cli import main
from rummage.store import Item, Segment, Store


def test_about_gathers_names_and_builds_a_timeline(tmp_path, capsys):
    store = Store(tmp_path / "data" / "index.db")
    now = time.time()
    rows = [
        ("notes", "people/selina.md", "note", "Selina Li", now - 9000, {"aliases": ["Selina", "大猫"]},
         "Selina Li is Livia's partner."),
        ("notes", "trip.md", "note", "Trip", now - 5000, {}, "大猫 booked the train tickets to Montreal."),
        ("chats", "s1:7", "chat", "Selina, chat", now - 100, {"speaker": "Selina", "session": "s1"},
         "Selina: can you check the pencil order?"),
        ("notes", "other.md", "note", "Other", now - 50, {}, "Nothing about anyone here."),
    ]
    for source, key, kind, title, mtime, info, text in rows:
        store.put(Item(source, key, kind, title, key, mtime, None, info), [Segment("text", text, loc="0")])
    store.commit()
    store.close()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n[people.'Selina Li']\naliases = ['Eosolene']\n", encoding="utf-8")
    assert main(["--config", str(cfg), "about", "Selina Li", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc["names"]) >= {"Selina Li", "Selina", "大猫", "Eosolene"}
    assert [r["ref"] for r in doc["results"]] == ["s1:7", "trip.md", "people/selina.md"]
    assert doc["results"][0]["how"] == "wrote"
