
import pytest

from qasr_cli.errors import UsageError
from qasr_cli.inputs import (
    expand_inputs,
    safe_output_path,
    sanitize_id,
    sanitize_rel_path,
)


def test_expand_single_file(wav_file):
    result = expand_inputs([str(wav_file)])
    assert len(result.items) == 1
    assert result.items[0].id == "clip"
    assert result.items[0].path == wav_file.resolve()


def test_expand_directory_recursive(tmp_path, wav_file):
    root = tmp_path / "audio"
    root.mkdir()
    (root / "a.wav").write_bytes(wav_file.read_bytes())
    sub = root / "sub"
    sub.mkdir()
    (sub / "b.wav").write_bytes(wav_file.read_bytes())
    (root / "notes.txt").write_text("ignore me")

    flat = expand_inputs([str(root)])
    assert [i.id for i in flat.items] == ["a"]

    recursive = expand_inputs([str(root)], recursive=True)
    assert sorted(i.id for i in recursive.items) == ["a", "sub/b"]


def test_expand_ignores_non_audio(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "x.txt").write_text("nope")
    result = expand_inputs([str(root)])
    assert result.items == []
    assert result.errors


def test_expand_glob(tmp_path, wav_file):
    root = tmp_path / "globbed"
    root.mkdir()
    (root / "a.wav").write_bytes(wav_file.read_bytes())
    (root / "b.wav").write_bytes(wav_file.read_bytes())
    result = expand_inputs([str(root / "*.wav")])
    assert sorted(i.id for i in result.items) == ["a", "b"]


def test_expand_missing_path_records_error(tmp_path):
    result = expand_inputs([str(tmp_path / "missing.wav")])
    assert result.items == []
    assert result.errors[0]["error"].exit_code == 4


def test_dedupe_by_absolute_path(wav_file):
    result = expand_inputs([str(wav_file), str(wav_file)])
    assert len(result.items) == 1


def test_duplicate_ids_are_disambiguated(tmp_path, wav_file):
    first = tmp_path / "a" / "clip.wav"
    second = tmp_path / "b" / "clip.wav"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(wav_file.read_bytes())
    second.write_bytes(wav_file.read_bytes())
    result = expand_inputs([str(first), str(second)])
    assert sorted(i.id for i in result.items) == ["clip", "clip_2"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello world", "hello_world"),
        ("../../etc/passwd", "etc_passwd"),
        ("a/b/../c", "a_b_c"),
        ("中文 名称", "中文_名称"),
        ("", "item"),
    ],
)
def test_sanitize_id(raw, expected):
    assert sanitize_id(raw) == expected


def test_sanitize_rel_path_rejects_escape():
    assert sanitize_rel_path("sub/a") == "sub/a"
    with pytest.raises(UsageError):
        sanitize_rel_path("../a")
    with pytest.raises(UsageError):
        sanitize_rel_path("/abs")


def test_safe_output_path(tmp_path):
    target = safe_output_path(tmp_path, "sub/x.srt")
    assert target == (tmp_path / "sub" / "x.srt").resolve()
    with pytest.raises(UsageError):
        safe_output_path(tmp_path, "../x.srt")
    with pytest.raises(UsageError):
        safe_output_path(tmp_path, "/etc/passwd")


def test_offline_url_not_cached(isolated_home):
    result = expand_inputs(["https://example.com/a.wav"], offline=True)
    assert result.errors
    assert "not cached" in result.errors[0]["error"].message
