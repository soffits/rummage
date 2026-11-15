import json

import numpy as np

from rummage.cli import main
from rummage.runtime import DIM
from rummage.store import Item, Segment, Store
from rummage.upkeep import facts


def _vec(seed, noise=0.0, base=None):
    rng = np.random.default_rng(seed)
    v = (base if base is not None else rng.normal(size=DIM) / np.sqrt(DIM)) + noise * rng.normal(size=DIM) / np.sqrt(DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_facts_pick_dates_times_amounts():
    assert facts("Dentist on 2026-11-18 at 3 pm, copay $25.50") == {"2026-11-18", "3 pm", "$25.50"}


def test_dupes_and_conflicts_reports(tmp_path, capsys):
    store = Store(tmp_path / "data" / "index.db")
    lease = _vec(1)
    trip = _vec(2)
    docs = [
        ("lease.md", "Rent is $2,150 a month, due on the 1st; the lease ends 2027-08-31.", _vec(10, 0.05, lease)),
        ("lease-copy.md", "Rent is $2,150 a month, due on the 1st; the lease ends 2027-08-31.", _vec(11, 0.01, lease)),
        ("old-lease-note.md", "Rent is $1,980 a month; the lease ends 2026-08-31.", _vec(12, 0.35, lease)),
        ("trip.md", "The train leaves at 9:00 on Friday.", trip),
    ]
    for key, text, vec in docs:
        store.put(Item("notes", key, "note", key, key), [Segment("text", text, loc="0", vector=vec)])
    store.commit()
    store.close()
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"data_dir = '{tmp_path / 'data'}'\n", encoding="utf-8")

    assert main(["--config", str(cfg), "dupes", "--json"]) == 0
    pairs = json.loads(capsys.readouterr().out)["results"]
    assert len(pairs) == 1 and {pairs[0]["a"]["ref"], pairs[0]["b"]["ref"]} == {"lease.md", "lease-copy.md"}

    assert main(["--config", str(cfg), "conflicts", "--json", "--min-cosine", "0.5"]) == 0
    found = json.loads(capsys.readouterr().out)["results"]
    refs = [{f["a"]["ref"], f["b"]["ref"]} for f in found]
    assert {"old-lease-note.md", "lease.md"} in refs or {"old-lease-note.md", "lease-copy.md"} in refs
    assert all("trip.md" not in r for r in refs)
    first = next(f for f in found if "old-lease-note.md" in {f["a"]["ref"], f["b"]["ref"]})
    old = first["a"] if first["a"]["ref"] == "old-lease-note.md" else first["b"]
    assert "$1,980" in old["facts"]
