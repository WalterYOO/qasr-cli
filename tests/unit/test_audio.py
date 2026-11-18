import pytest

from qasr_cli.audio import max_new_tokens_for, probe
from qasr_cli.errors import InputError
from tests.conftest import write_wav


def test_probe_wav(tmp_path):
    path = write_wav(tmp_path / "a.wav", seconds=2.0, sr=16000)
    info = probe(path)
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.duration_s == pytest.approx(2.0, abs=1e-3)
    assert not info.is_empty


def test_probe_empty_wav(tmp_path):
    path = write_wav(tmp_path / "empty.wav", seconds=0)
    info = probe(path)
    assert info.is_empty
    assert info.duration_s == 0.0


def test_probe_missing(tmp_path):
    with pytest.raises(InputError, match="not found"):
        probe(tmp_path / "missing.wav")


def test_probe_corrupt(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"not audio at all")
    with pytest.raises(InputError, match="cannot read audio"):
        probe(path)


@pytest.mark.parametrize(
    "duration,expected",
    [(0, 1024), (30, 1024), (100, 1056), (2000, 16256), (5000, 16384)],
)
def test_max_new_tokens(duration, expected):
    assert max_new_tokens_for(duration) == expected
