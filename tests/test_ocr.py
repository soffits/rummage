"""Real OCR on pictures of text in each CJK language and in English (needs the `ocr` extra; the models download once
into the data folder). The pictures are lines set in Noto Sans SC, TC, JP, KR and Noto Sans."""

from pathlib import Path

import pytest

pytest.importorskip("rapidocr")
pytest.importorskip("onnxruntime")

from rummage import ocr as O  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "ocr"


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    r = O.Reader(tmp_path_factory.mktemp("ocr-data"))
    yield r
    r.release()


@pytest.mark.parametrize("picture, words", [
    ("sc.png", ["租赁合同第三条", "租金每月一千二百元", "2024年8月1日"]),        # Simplified Chinese
    ("tc.png", ["臺灣銀行存款證明書", "林曉華", "0123456789"]),                  # Traditional Chinese
    ("ja.png", ["在留カードの更新手続きについて", "住民票の写し"]),              # Japanese
    ("ko.png", ["대한민국 여권 발급 신청서", "서울특별시 강남구 주민센터"]),     # Korean, read by the second model
    ("en.png", ["Ontario Driver's Licence Renewal", "INV-20481", "$1,284.50"]),
])
def test_reads_cjk_and_latin(reader, picture, words):
    found = reader.text(None, (FIXTURES / picture).read_bytes())
    for word in words:
        assert word in found.replace(" ", "") or word in found, (word, found)


def test_cache_and_undecodable(reader):
    first = reader.text("k1", (FIXTURES / "en.png").read_bytes())
    assert reader.text("k1", b"not read again") == first
    with pytest.raises(O.Undecodable):
        reader.text(None, b"not a picture")


def test_idle_ocr_threads_sleep(reader):
    # Each model's thread pool would otherwise spin after its run, using cores beside the reading run's share.
    reader.text(None, (FIXTURES / "en.png").read_bytes())
    for engine in reader._load():
        for part in (engine.text_det, engine.text_rec):
            if part is not None:
                options = part.session.session.get_session_options()
                assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
