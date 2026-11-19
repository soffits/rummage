import datetime as dt
import json

from rummage.cli import main
from rummage.sources.folder import path_dates
from rummage.store import Item, Segment, Store


def _day(s):
    return dt.datetime.fromisoformat(s).timestamp()


def test_dated_note_names_count_as_their_time():
    assert path_dates("weather/2026-10-15.md")["when"] == _day("2026-10-15")
    assert path_dates("weekly/2026-W44.md")["when"] == _day("2026-10-26")
    assert path_dates("finance/months/2026-10.md")["when_end"] == _day("2026-11-01")
    assert path_dates("notes/lease.md") == {} and path_dates("ids/4815162342.md") == {}


def _index(tmp_path):
    """Three items about the same plan: a weekly note for W45 edited much later, one plain note from September,
    one from November."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n", encoding="utf-8")
    store = Store(tmp_path / "data" / "index.db")
    for key, mtime, info in (("weekly/2026-W45.md", "2026-12-20", path_dates("weekly/2026-W45.md")),
                             ("old-plan.md", "2026-09-02", {}), ("new-plan.md", "2026-11-20", {})):
        store.put(Item("notes", key, "note", key, key, _day(mtime) + 3600, None, info),
                  [Segment("text", "the weekly plan and its open tasks", loc="0")])
    store.commit()
    store.close()
    return cfg


def _refs(capsys, cfg, *args):
    assert main(["--config", str(cfg), "search", "weekly plan tasks", "--words-only", "--json", *args]) == 0
    return [r["ref"] for r in json.loads(capsys.readouterr().out)["results"]]


def test_since_and_until_include_the_period_a_note_is_about(tmp_path, capsys):
    cfg = _index(tmp_path)
    assert sorted(_refs(capsys, cfg, "--since", "2026-11-02", "--until", "2026-11-08")) == ["weekly/2026-W45.md"]
    assert sorted(_refs(capsys, cfg, "--until", "2026-10-01")) == ["old-plan.md"]


def test_newest_first(tmp_path, capsys):
    cfg = _index(tmp_path)
    assert _refs(capsys, cfg, "--sort", "newest") == ["weekly/2026-W45.md", "new-plan.md", "old-plan.md"]


def test_oldest_first(tmp_path, capsys):
    cfg = _index(tmp_path)
    assert _refs(capsys, cfg, "--sort", "oldest") == ["old-plan.md", "new-plan.md", "weekly/2026-W45.md"]
