from rummage import text as T
from rummage.recall import worth_recalling
from rummage.search import Searcher
from rummage.store import Item, Segment, Store


def _index(tmp_path, docs):
    store = Store(tmp_path / "index.db")
    for key, body in docs.items():
        store.put(Item("notes", key, "note", key, key), [Segment("text", body, loc="0")])
    store.commit()
    return store


def _top(store, query):
    hits = Searcher(store.db).search(query, None, limit=3)
    return hits[0].ref if hits else None


def test_terms_cover_all_cjk_scripts():
    assert T.terms("宠物") == ["宠物"]
    assert T.terms("東京で寿司") == ["東京", "京で", "で寿", "寿司"]
    assert T.terms("학교에서") == ["학교", "교에", "에서"]
    assert T.terms("ｶﾀｶﾅ") == T.terms("カタカナ")
    assert T.terms("ＡＢＣ１２３") == ["abc123"]
    assert T.for_index("한국어").split() == ["한", "국", "어"]


def test_keyword_search_in_chinese_japanese_korean(tmp_path):
    store = _index(tmp_path, {
        "zh.md": "我们决定下个月搬家，新公寓可以养猫。",
        "ja.md": "昨日は東京で寿司を食べました。とても美味しかったです。",
        "ko.md": "내일 학교에 가서 시험을 봐요. 도서관에서 공부했어요.",
        "en.md": "Notes about the weekend trip and the hiking boots.",
    })
    assert _top(store, "养猫") == "zh.md"
    assert _top(store, "東京の寿司") == "ja.md"
    assert _top(store, "학교에서 시험") == "ko.md"
    assert _top(store, "도서관") == "ko.md"
    store.close()


def test_half_width_katakana_in_documents(tmp_path):
    store = _index(tmp_path, {"kana.md": "ﾃﾞｰﾀﾍﾞｰｽの設定を変更した", "other.md": "nothing related here at all"})
    assert _top(store, "データベース") == "kana.md"
    store.close()


def test_recall_threshold_counts_any_cjk_script():
    assert worth_recalling("租约能养猫吗")
    assert worth_recalling("東京の寿司屋")
    assert worth_recalling("학교 시험 일정")
    assert not worth_recalling("好")
