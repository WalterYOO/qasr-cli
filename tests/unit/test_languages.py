import pytest

from qasr_cli.errors import InputError
from qasr_cli.languages import (
    ALIGNABLE,
    DIALECTS,
    FORCEABLE,
    LID_LABELS,
    detect_script_language,
    languages_payload,
    resolve_alignable,
    resolve_forceable,
)


def test_table_sizes():
    assert len(FORCEABLE) == 30
    assert len(DIALECTS) == 22
    assert len(LID_LABELS) == 52
    assert len(ALIGNABLE) == 11


def test_payload_shape():
    payload = languages_payload()
    assert payload["schema_version"] == 1
    assert len(payload["forceable"]) == 30
    assert payload["alignable"][0] == "Chinese"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("zh", "Chinese"),
        ("Chinese", "Chinese"),
        ("cmn", "Chinese"),
        ("中文", "Chinese"),
        ("en", "English"),
        ("yue", "Cantonese"),
        ("JA", "Japanese"),
        ("mk", "Macedonian"),
    ],
)
def test_forceable_aliases(value, expected):
    assert resolve_forceable(value) == expected


def test_dialect_cannot_be_forced():
    with pytest.raises(InputError, match="dialect"):
        resolve_forceable("Sichuan")


def test_unknown_language():
    with pytest.raises(InputError, match="unsupported language"):
        resolve_forceable("klingon")


def test_auto_is_not_forceable():
    with pytest.raises(InputError):
        resolve_forceable("auto")


def test_alignable_rejects_non_alignable_forceable_language():
    assert resolve_alignable("zh") == "Chinese"
    with pytest.raises(InputError, match="cannot be aligned"):
        resolve_alignable("th")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("甚至出现交易几乎停滞的情况。", "Chinese"),
        ("これはテストです", "Japanese"),
        ("안녕하세요", "Korean"),
        ("Привет мир", "Russian"),
    ],
)
def test_script_detection(text, expected):
    assert detect_script_language(text) == expected


def test_script_detection_latin_requires_explicit():
    with pytest.raises(InputError, match="Latin"):
        detect_script_language("hello world")
