"""LANG-01..12: the three language tables and their CLI semantics."""

from __future__ import annotations

import pytest
from helpers import assert_error, assert_exit, parse_json


def _tables(qasr):
    proc = qasr.run("--json", "languages")
    assert_exit(proc, 0, context="languages --json")
    payload = parse_json(proc)
    forceable = payload.get("forceable")
    alignable = payload.get("alignable")
    dialects = payload.get("dialects")
    lid_labels = payload.get("lid_labels")
    assert isinstance(forceable, list), f"forceable is {type(forceable).__name__}"
    assert isinstance(alignable, list), f"alignable is {type(alignable).__name__}"
    assert isinstance(dialects, list), f"dialects is {type(dialects).__name__}"
    assert isinstance(lid_labels, list), f"lid_labels is {type(lid_labels).__name__}"
    return payload, forceable, alignable, dialects, lid_labels


def _names(entries):
    names = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict):
            names.append(entry.get("name"))
    return [name for name in names if isinstance(name, str)]


def _iso_map(forceable):
    mapping = {}
    for entry in forceable:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        iso = entry.get("iso")
        if isinstance(name, str) and isinstance(iso, str):
            mapping[iso.lower()] = name
    return mapping


def test_forceable_has_30_languages(qasr):
    _, forceable, _, _, _ = _tables(qasr)
    assert len(forceable) == 30, f"expected 30 forceable languages, got {len(forceable)}"


def test_alignable_has_11_languages_and_is_subset(qasr):
    _, forceable, alignable, _, _ = _tables(qasr)
    assert len(alignable) == 11, f"expected 11 alignable languages, got {alignable!r}"
    forceable_names = {name.casefold() for name in _names(forceable)}
    alignable_names = _names(alignable)
    assert alignable_names, f"alignable entries are not names: {alignable!r}"
    missing = [name for name in alignable_names if name.casefold() not in forceable_names]
    assert not missing, f"alignable languages not forceable: {missing}"


def test_dialects_and_lid_counts(qasr):
    _, forceable, alignable, dialects, lid_labels = _tables(qasr)
    assert len(dialects) == 22, f"expected 22 dialects, got {len(dialects)}"
    assert len(lid_labels) == 52, f"expected 30+22=52 LID labels, got {len(lid_labels)}"
    assert len(forceable) + len(dialects) == len(lid_labels)


def test_dialects_are_not_forceable(qasr):
    _, forceable, _, dialects, lid_labels = _tables(qasr)
    forceable_names = {name.casefold() for name in _names(forceable)}
    dialect_names = {name.casefold() for name in _names(dialects)}
    assert "sichuan" in dialect_names
    assert "sichuan" not in forceable_names
    lid_names = {name.casefold() for name in _names(lid_labels)}
    assert "sichuan" in lid_names


def test_iso_aliases(qasr):
    _, forceable, _, _, _ = _tables(qasr)
    mapping = _iso_map(forceable)
    expected = {
        "zh": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "pt": "Portuguese",
        "it": "Italian",
    }
    for iso, name in expected.items():
        assert mapping.get(iso) == name, f"iso {iso!r} maps to {mapping.get(iso)!r}, want {name!r}"
    for entry in forceable:
        assert isinstance(entry, dict), f"forceable entry is not an object: {entry!r}"
        assert "aliases" in entry, f"forceable entry has no aliases: {entry!r}"


def test_dialect_rejected_by_language_flag(qasr, make_wav):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run("--json", "--dry-run", "transcribe", audio, "--language", "Sichuan")
    assert_exit(proc, 4, context="force dialect")
    error = assert_error(parse_json(proc), code=4)
    text = (error.get("message", "") + " " + str(error.get("hint", ""))).lower()
    assert "auto" in text or "dialect" in text, f"dialect error lacks guidance: {error!r}"


@pytest.mark.parametrize("language", ["zh", "Chinese", "yue", "en"])
def test_accepted_language_forms_plan(qasr, make_wav, language):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run("--json", "--dry-run", "transcribe", audio, "--language", language)
    assert_exit(proc, 0, context=f"--language {language}")
    payload = parse_json(proc)
    assert payload.get("ok") is True


def test_unknown_language_is_rejected(qasr, make_wav):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run("--json", "--dry-run", "transcribe", audio, "--language", "klingon")
    assert_exit(proc, {2, 4}, context="unknown language")
    payload = parse_json(proc)
    assert_error(payload)


def test_align_rejects_forceable_but_unalignable_language(qasr, make_wav):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "align",
        audio,
        "--text",
        "hello",
        "--language",
        "th",
    )
    assert_exit(proc, 4, context="Thai is forceable but not alignable")
    assert_error(parse_json(proc), code=4)


def test_align_auto_accepts_han_script(qasr, make_wav):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "align",
        audio,
        "--text",
        "你好世界",
        "--language",
        "auto",
    )
    # Passing the script gate is proven by reaching model resolution (exit 3),
    # not by a usage/capability rejection.
    assert_exit(proc, 3, context="align auto with Han text")


def test_align_auto_rejects_latin_script(qasr, make_wav):
    audio = make_wav(qasr.work / "a.wav", seconds=0.5)
    proc = qasr.run(
        "--json",
        "--model",
        "/nonexistent/model",
        "align",
        audio,
        "--text",
        "hello world",
        "--language",
        "auto",
    )
    assert_exit(proc, 4, context="align auto with Latin text")
    error = assert_error(parse_json(proc), code=4)
    text = (error.get("message", "") + " " + str(error.get("hint", ""))).lower()
    assert "language" in text, f"Latin auto error lacks language guidance: {error!r}"
