"""The three language tables required by the design contract.

1. Forceable languages (30) — the only values accepted by ``--language``.
2. LID labels (30 canonical + 22 Chinese dialects) — model output, never validated.
3. Alignable languages (11) — accepted by ``--timestamps`` and ``qasr align``.

This module is pure Python and must never import torch/qwen_asr.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .errors import InputError


@dataclass(frozen=True)
class Language:
    name: str
    iso: str
    aliases: tuple[str, ...] = field(default=())


# Canonical names must match qwen_asr.inference.utils.SUPPORTED_LANGUAGES exactly.
FORCEABLE: tuple[Language, ...] = (
    Language("Chinese", "zh", ("zho", "chi", "cmn", "mandarin", "中文", "汉语", "普通话")),
    Language("English", "en", ("eng", "英语", "英文")),
    Language("Cantonese", "yue", ("zh-yue", "cantonese", "粤语", "广东话")),
    Language("Arabic", "ar", ("ara", "阿拉伯语")),
    Language("German", "de", ("deu", "ger", "德语")),
    Language("French", "fr", ("fra", "fre", "法语")),
    Language("Spanish", "es", ("spa", "西班牙语")),
    Language("Portuguese", "pt", ("por", "葡萄牙语")),
    Language("Indonesian", "id", ("ind", "印尼语")),
    Language("Italian", "it", ("ita", "意大利语")),
    Language("Korean", "ko", ("kor", "韩语")),
    Language("Russian", "ru", ("rus", "俄语")),
    Language("Thai", "th", ("tha", "泰语")),
    Language("Vietnamese", "vi", ("vie", "越南语")),
    Language("Japanese", "ja", ("jpn", "日语")),
    Language("Turkish", "tr", ("tur", "土耳其语")),
    Language("Hindi", "hi", ("hin", "印地语")),
    Language("Malay", "ms", ("msa", "may", "马来语")),
    Language("Dutch", "nl", ("nld", "dut", "荷兰语")),
    Language("Swedish", "sv", ("swe", "瑞典语")),
    Language("Danish", "da", ("dan", "丹麦语")),
    Language("Finnish", "fi", ("fin", "芬兰语")),
    Language("Polish", "pl", ("pol", "波兰语")),
    Language("Czech", "cs", ("ces", "cze", "捷克语")),
    Language("Filipino", "fil", ("tl", "tgl", "菲律宾语")),
    Language("Persian", "fa", ("fas", "per", "波斯语")),
    Language("Greek", "el", ("ell", "gre", "希腊语")),
    Language("Romanian", "ro", ("ron", "rum", "罗马尼亚语")),
    Language("Hungarian", "hu", ("hun", "匈牙利语")),
    Language("Macedonian", "mk", ("mkd", "mac", "马其顿语")),
)

DIALECTS: tuple[str, ...] = (
    "Anhui",
    "Dongbei",
    "Fujian",
    "Gansu",
    "Guizhou",
    "Hebei",
    "Henan",
    "Hubei",
    "Hunan",
    "Jiangxi",
    "Ningxia",
    "Shandong",
    "Shaanxi",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Yunnan",
    "Zhejiang",
    "Cantonese (Hong Kong)",
    "Cantonese (Guangdong)",
    "Wu",
    "Minnan",
)

ALIGNABLE: tuple[str, ...] = (
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)

FORCEABLE_NAMES: tuple[str, ...] = tuple(lang.name for lang in FORCEABLE)
LID_LABELS: tuple[str, ...] = FORCEABLE_NAMES + DIALECTS

ALIGN_HINT = "pass --language " + "|".join(lang.iso for lang in FORCEABLE if lang.name in ALIGNABLE)

_ALIAS_TO_NAME: Dict[str, str] = {}
_NAME_BY_CASEFOLD: Dict[str, str] = {}
for _lang in FORCEABLE:
    _NAME_BY_CASEFOLD[_lang.name.casefold()] = _lang.name
    _ALIAS_TO_NAME.setdefault(_lang.name.casefold(), _lang.name)
    _ALIAS_TO_NAME.setdefault(_lang.iso.casefold(), _lang.name)
    for _alias in _lang.aliases:
        _ALIAS_TO_NAME.setdefault(_alias.casefold(), _lang.name)

_DIALECT_BY_CASEFOLD = {d.casefold(): d for d in DIALECTS}


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def resolve_forceable(value: str) -> str:
    """Map an ISO code/alias to a canonical forceable language name."""
    if value is None or not str(value).strip():
        raise InputError("language must not be empty")
    key = _nfc(str(value)).casefold()
    if key in ("auto", ""):
        raise InputError("'auto' is not a forceable language; omit --language for LID")
    if key in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[key]
    if key in _DIALECT_BY_CASEFOLD:
        raise InputError(
            f"'{value}' is a dialect label and cannot be forced",
            hint="use --language auto; dialects are only emitted by LID",
        )
    raise InputError(
        f"unsupported language: {value}",
        hint="run 'qasr languages' to list the 30 forceable languages",
        details={"forceable": list(FORCEABLE_NAMES)},
    )


def is_alignable(name: str) -> bool:
    return name in ALIGNABLE


def resolve_alignable(value: str) -> str:
    """Resolve an ISO code/alias to one of the 11 alignable canonical names."""
    name = resolve_forceable(value)
    if name not in ALIGNABLE:
        raise InputError(
            f"language '{name}' cannot be aligned",
            hint="alignable: " + ", ".join(ALIGNABLE),
            details={"alignable": list(ALIGNABLE)},
        )
    return name


def iso_for(name: str) -> Optional[str]:
    for lang in FORCEABLE:
        if lang.name == name:
            return lang.iso
    return None


def forceable_table() -> List[dict]:
    return [
        {"name": lang.name, "iso": lang.iso, "aliases": list(lang.aliases)}
        for lang in FORCEABLE
    ]


def languages_payload() -> dict:
    from . import SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "command": "languages",
        "forceable": forceable_table(),
        "lid_labels": list(LID_LABELS),
        "alignable": list(ALIGNABLE),
        "dialects": list(DIALECTS),
    }


def detect_script_language(text: str) -> str:
    """P0 ``align --language auto``: script-based detection.

    Latin and other scripts are rejected on purpose (French/German would be
    indistinguishable from English without a model).
    """
    has_han = has_kana = has_hangul = has_cyrillic = has_latin = False
    for ch in text:
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            has_kana = True
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            has_han = True
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            has_hangul = True
        elif 0x0400 <= code <= 0x04FF:
            has_cyrillic = True
        elif ("a" <= ch.lower() <= "z"):
            has_latin = True

    if has_kana:
        return "Japanese"
    if has_hangul:
        return "Korean"
    if has_han:
        return "Chinese"
    if has_cyrillic:
        return "Russian"
    if has_latin:
        raise InputError(
            "cannot infer an alignment language from Latin script",
            hint="pass --language explicitly (e.g. --language en|fr|de|es|it|pt)",
        )
    raise InputError(
        "cannot infer an alignment language from this text",
        hint="pass --language explicitly",
    )
