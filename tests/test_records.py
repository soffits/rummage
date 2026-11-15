import datetime as dt
import json

from conftest import DIM, index_all
from rummage import records as R
from rummage.sources.files import FilesSource
from rummage.store import Store

import numpy as np


class _Model:
    def texts(self, texts):
        return np.ones((len(texts), DIM), np.float32) / np.sqrt(DIM)


ORDERS = {"title": "Orders", "records": [
    {"orderId": "A-1", "purchaseDate": "2026-03-04", "items": [{"title": "Brass hinge set (4)", "qty": 1}]},
    {"orderId": "A-2", "purchaseDate": "2026-03-09", "items": [{"title": "Lithium coin cell CR9999", "qty": 5}]},
]}


def test_json_array_under_a_key_gives_one_record_each():
    found = R.parse("orders.json", json.dumps(ORDERS).encode())
    assert [r.title for r in found] == ["Brass hinge set (4)", "Lithium coin cell CR9999"]
    assert "items.title: Lithium coin cell CR9999" in found[1].text and "orderId: A-2" in found[1].text
    first, second = (dt.datetime.fromtimestamp(r.when).date() for r in found)
    assert (first, second) == (dt.date(2026, 3, 4), dt.date(2026, 3, 9))
    assert dt.datetime.fromtimestamp(found[0].when_end).date() == dt.date(2026, 3, 5)


def test_plain_array_jsonl_csv_and_tsv():
    rows = [{"name": "Ana", "note": "likes tea"}, {"name": "Bo", "note": "likes jam"}]
    assert [r.title for r in R.parse("x.json", json.dumps(rows).encode())] == ["Ana", "Bo"]
    assert [r.title for r in R.parse("x.jsonl", "\n".join(map(json.dumps, rows)).encode())] == ["Ana", "Bo"]
    assert [r.title for r in R.parse("x.csv", b"name,note\nAna,likes tea\nBo,likes jam\n")] == ["Ana", "Bo"]
    tsv = R.parse("x.tsv", "subject\tbody\nPlan\tbuy bread\nTrip\tpack bags\n".encode())
    assert [r.title for r in tsv] == ["Plan", "Trip"] and "body: pack bags" in tsv[1].text


def test_title_falls_back_to_first_text_and_ids_are_not_titles():
    found = R.parse("x.csv", b"id,place\n10,Lisbon\n11,Porto\n")
    assert [r.title for r in found] == ["Lisbon", "Porto"]


def test_data_that_is_not_record_shaped_stays_a_file():
    assert R.parse("c.json", json.dumps({"a": 1, "b": {"c": 2}}).encode()) is None
    assert R.parse("c.json", json.dumps([1, 2, 3]).encode()) is None
    assert R.parse("c.json", json.dumps([{"a": "only one"}]).encode()) is None
    assert R.parse("c.json", b"{broken") is None
    assert R.parse("c.csv", b"single,header\n") is None


def test_records_are_indexed_as_items_and_follow_edits(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "orders.json").write_text(json.dumps(ORDERS))
    (root / "settings.json").write_text(json.dumps({"theme": "dark"}))
    store = Store(tmp_path / "index.db")
    source = FilesSource("docs", {"root": str(root)}, tmp_path)
    counts = index_all(store, source, _Model(), lambda *a: None, print)
    assert counts["added"] == 3

    def titles():
        return sorted(r[0] for r in store.db.execute("SELECT title FROM items"))

    assert titles() == ["Brass hinge set (4)", "Lithium coin cell CR9999", "settings"]
    hit = store.db.execute("SELECT i.title FROM segments s JOIN items i ON i.id = s.item "
                           "WHERE s.text LIKE '%CR9999%'").fetchone()
    assert hit[0] == "Lithium coin cell CR9999"

    ORDERS["records"].insert(0, {"orderId": "A-0", "items": [{"title": "Cork mat"}]})
    (root / "orders.json").write_text(json.dumps(ORDERS))
    counts = index_all(store, source, _Model(), lambda *a: None, print)
    assert counts["added"] == 1 and counts["removed"] == 0
    ORDERS["records"].pop(1)
    (root / "orders.json").write_text(json.dumps(ORDERS))
    counts = index_all(store, source, _Model(), lambda *a: None, print)
    assert counts["removed"] == 1 and "Brass hinge set (4)" not in titles()
    key = store.db.execute("SELECT key FROM items WHERE title = 'Cork mat'").fetchone()[0]
    assert "Cork mat" in source.original(key)["text"]
    store.close()


def test_titles_skip_dates_and_numbers():
    found = R.parse("x.csv", b"order_time,qty,item\n2026-01-02 10:00:00,2,Tea tin\n2026-01-03 11:00:00,1,Jam jar\n")
    assert [r.title for r in found] == ["Tea tin", "Jam jar"]
    found = R.parse("y.csv", b"when,count,place\n2026-01-02,3,Lisbon\n2026-01-04,4,Porto\n")
    assert [r.title for r in found] == ["Lisbon", "Porto"]
