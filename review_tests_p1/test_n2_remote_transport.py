"""N2: remote transport contract (transcription and chat)."""

from __future__ import annotations

import json
import os
import shutil

import pytest
from fake_openai import error_response, json_response
from helpers import assert_exit, parse_json, validate_schema


def _find_audio_part(body: dict):
    messages = body.get("messages") or []
    assert messages, f"chat body has no messages: {body!r}"
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"audio_url", "input_audio"}:
                return part
    return None


def test_transcription_multipart_contract(qasr, fake_openai, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>远程文本", "usage": {"seconds": 1}}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav, "--json"
    )
    assert_exit(proc, 0, context="remote transcription")
    payload = parse_json(proc)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["result"]["text"] == "远程文本"
    assert str(payload["result"]["language"]).lower() in {"chinese", "zh", "中文"}
    assert payload["result"]["language_source"] == "server"
    validate_schema(payload, "transcribe-item.json")

    posts = server.requests_for("/v1/audio/transcriptions")
    assert len(posts) == 1, [request.path for request in server.post_requests()]
    fields, files = posts[0].form
    assert "file" in files, fields
    assert files["file"].filename == short_wav.name
    assert "wav" in files["file"].content_type.lower()
    assert fields.get("model") == "Qwen/Qwen3-ASR-1.7B"
    assert "language" not in fields or fields["language"] in {"", "auto"}


def test_transcription_plain_text_has_null_language(qasr, fake_openai, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response({"text": "纯文本"})
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 0, context="remote plain text")
    result = parse_json(proc)["result"]
    assert result["text"] == "纯文本"
    assert result["language"] is None
    assert result["language_source"] == "server"


@pytest.mark.parametrize(
    ("raw", "language", "text"),
    [
        ("language Chinese,English<asr_text>你好", "Chinese,English", "你好"),
        ("language Chinese\nmetadata: 1\n<asr_text>你好", "Chinese", "你好"),
        ("language None<asr_text>你好", None, "你好"),
        ("language zh\n纯文本", "zh", "纯文本"),
    ],
)
def test_language_prefix_variants_are_parsed(
    qasr, fake_openai, short_wav, raw, language, text
):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response({"text": raw})
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 0, context="language prefix")
    result = parse_json(proc)["result"]
    assert result["language"] == language
    assert result["text"] == text
    assert "<asr_text>" not in result["text"]
    assert result["language_source"] == "server"


def test_empty_remote_output_is_success(qasr, fake_openai, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response({"text": ""})
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 0, context="empty remote output")
    result = parse_json(proc)["result"]
    assert result["text"] == ""
    assert result["language"] is None
    assert result["language_source"] == "server"


def test_explicit_language_is_sent_as_iso_code(qasr, fake_openai, short_wav):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--language", "zh", "--json",
    )
    assert_exit(proc, 0, context="remote language zh")
    fields, _files = server.requests_for("/v1/audio/transcriptions")[0].form
    assert fields.get("language") == "zh"


def test_server_language_is_not_overwritten_by_client(qasr, fake_openai, short_wav):
    """gpu06 fact: legal --language does not change server LID output."""
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>服务端识别为中文"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--language", "en", "--json",
    )
    assert_exit(proc, 0, context="remote language mismatch")
    result = parse_json(proc)["result"]
    assert str(result["language"]).lower() in {"chinese", "zh", "中文"}
    warnings = result.get("warnings") or []
    assert warnings, "explicit language mismatch must produce a warning"
    assert any("language" in str(item).lower() or "语言" in str(item) for item in warnings)


def test_explicit_served_model_is_used_verbatim(qasr, fake_openai, short_wav):
    server = fake_openai(models=["first-model", "second-model"])
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--model", "second-model", "--json",
    )
    assert_exit(proc, 0, context="remote model override")
    fields, _files = server.requests_for("/v1/audio/transcriptions")[0].form
    assert fields.get("model") == "second-model"


def test_unknown_model_lists_served_models(qasr, fake_openai, short_wav):
    server = fake_openai(models=["first-model", "second-model"])
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--model", "remote-custom", "--json",
    )
    assert_exit(proc, 4, context="unknown remote model")
    payload = parse_json(proc)
    assert payload["error"]["code"] == 4
    hint = payload["error"].get("hint") or ""
    assert "first-model" in hint and "second-model" in hint, hint
    assert not server.post_requests(), "model resolution must precede upload"


def test_default_model_is_first_models_entry(qasr, fake_openai, short_wav):
    server = fake_openai(models=["first-model", "second-model"])
    proc = qasr.run("--server", server.url, "transcribe", short_wav, "--json")
    assert_exit(proc, 0, context="remote default model")
    fields, _files = server.requests_for("/v1/audio/transcriptions")[0].form
    assert fields.get("model") == "first-model"


def test_chat_context_contract(qasr, fake_openai, short_wav):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--context", "交易", "--context", "停滞", "--json",
    )
    assert_exit(proc, 0, context="remote chat context")
    payload = parse_json(proc)
    assert payload["result"]["text"] == "远程文本"
    assert str(payload["result"]["language"]).lower() in {"chinese", "zh", "中文"}
    posts = server.requests_for("/v1/chat/completions")
    assert len(posts) == 1, [request.path for request in server.post_requests()]
    body = posts[0].json
    assert isinstance(body, dict), posts[0].body[:500]
    messages = body.get("messages")
    assert isinstance(messages, list) and messages
    system = messages[0]
    assert system.get("role") == "system"
    assert "交易" in json.dumps(system, ensure_ascii=False)
    assert "停滞" in json.dumps(system, ensure_ascii=False)
    part = _find_audio_part(body)
    assert part is not None, f"no audio part in chat body: {body!r}"
    if part.get("type") == "audio_url":
        url = part["audio_url"]["url"]
    else:
        url = part["input_audio"]["data"]
        if not url.startswith("data:"):
            url = "data:audio/wav;base64," + url
    assert url.startswith("data:audio/")
    assert ";base64," in url


def test_chat_explicit_language_does_not_require_prefill(qasr, fake_openai, short_wav):
    """gpu06 fact: assistant prefill + continue_final_message always 400."""
    server = fake_openai()

    def handler(request, srv):
        if request.path.split("?", 1)[0] == "/v1/chat/completions":
            body = request.json or {}
            roles = [message.get("role") for message in body.get("messages", [])]
            if "assistant" in roles or body.get("continue_final_message"):
                return error_response("prefill is not supported", status=400)
            return srv.chat_response(request, srv)
        return None

    server.handler = handler
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--context", "交易", "--language", "zh", "--json",
    )
    assert_exit(proc, 0, context="chat without prefill")
    payload = parse_json(proc)
    assert payload["result"]["text"] == "远程文本"
    assert payload["result"].get("warnings"), "chat cannot enforce language; must warn"
    posts = server.requests_for("/v1/chat/completions")
    assert len(posts) == 1, (
        "client must not send assistant prefill when the server rejects it; "
        f"observed {len(posts)} chat requests"
    )


def test_transcription_language_rejection_retries_without_language_once(
    qasr, fake_openai, short_wav
):
    server = fake_openai()
    attempts = []

    def handler(request, srv):
        if request.path.split("?", 1)[0] != "/v1/audio/transcriptions":
            return None
        fields, _files = request.form
        attempts.append(fields.get("language"))
        if fields.get("language"):
            return error_response("Unsupported language: zh", status=400)
        return json_response({"text": "language Chinese<asr_text>成功"})

    server.handler = handler
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--language", "zh", "--retries", "2", "--json",
    )
    assert_exit(proc, 0, context="language rejection retry")
    payload = parse_json(proc)
    assert payload["result"]["text"] == "成功"
    assert attempts == ["zh", None], attempts
    assert payload["result"].get("warnings"), "de-language retry must warn"


def test_transcription_404_falls_back_to_chat_and_caches(qasr, fake_openai, tmp_path, short_wav):
    server = fake_openai()

    def handler(request, srv):
        path = request.path.split("?", 1)[0]
        if path == "/v1/audio/transcriptions":
            return error_response("not found", status=404)
        if path == "/v1/chat/completions":
            return srv.chat_response(request, srv)
        return None

    server.handler = handler
    second = tmp_path / "second.wav"
    shutil.copyfile(short_wav, second)
    out = tmp_path / "out"
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav, second,
        "--out-dir", out, "--jobs", "1", "--json",
    )
    assert_exit(proc, 0, context="404 fallback")
    payload = parse_json(proc)
    assert payload["summary"]["ok"] == 2
    assert len(server.requests_for("/v1/audio/transcriptions")) == 1
    assert len(server.requests_for("/v1/chat/completions")) == 2
    warnings = payload["items"][0]["result"].get("warnings") or []
    assert any("chat" in str(item).lower() for item in warnings), warnings


def test_remote_warnings_are_echoed_to_stderr(qasr, fake_openai, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>中文"}
    )
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--language", "en", "--json",
    )
    assert_exit(proc, 0, context="warning on stderr")
    assert "language" in proc.stderr.lower(), proc.stderr


def test_remote_does_not_import_torch(
    qasr, fake_openai, short_wav, import_blocker, repo_root
):
    server = fake_openai()
    blocker, marker = import_blocker
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav, "--json",
        env={"PYTHONPATH": os.pathsep.join([str(blocker), str(repo_root)])},
    )
    assert_exit(proc, 0, context="remote without torch")
    assert not marker.exists(), "remote transcription imported a local runtime"


def test_remote_timestamps_is_capability_error(qasr, fake_openai, short_wav):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--timestamps", "--language", "zh", "--json",
    )
    assert_exit(proc, 4, context="remote timestamps")
    payload = parse_json(proc)
    assert payload["error"]["kind"] == "capability"
    assert not server.post_requests(), "capability check must precede upload"


def test_remote_non_sentence_granularity_is_capability_error(
    qasr, fake_openai, short_wav
):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", short_wav,
        "--granularity", "word", "--json",
    )
    assert_exit(proc, 4, context="remote granularity")
    payload = parse_json(proc)
    assert payload["error"]["kind"] == "capability"
    assert not server.post_requests()


def test_remote_align_is_capability_error(qasr, fake_openai, short_wav):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "align", short_wav,
        "--text", "你好", "--language", "zh", "--json",
    )
    assert_exit(proc, 4, context="remote align")
    payload = parse_json(proc)
    assert payload["error"]["kind"] == "capability"
    assert not server.post_requests()


def test_remote_missing_input_fails_before_upload(qasr, fake_openai, tmp_path):
    server = fake_openai()
    proc = qasr.run(
        "--server", server.url, "transcribe", tmp_path / "missing.wav", "--json"
    )
    assert_exit(proc, 4, context="remote missing input")
    assert not server.post_requests()


def test_server_flag_after_subcommand(qasr, fake_openai, short_wav):
    server = fake_openai()
    server.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>flag 后置"}
    )
    proc = qasr.run(
        "transcribe", short_wav, "--server", server.url, "--json"
    )
    if proc.returncode != 0:
        pytest.fail(
            "--server must be accepted after the subcommand like other global flags\n"
            f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    assert parse_json(proc)["result"]["text"] == "flag 后置"


def test_remote_audio_url_input_downloads_first(qasr, fake_openai, tmp_path):
    """URL inputs go through the P0 URL cache before upload."""
    source = fake_openai(handler=lambda request, srv: None)
    source.transcription_response = lambda request, srv: json_response(
        {"text": "language Chinese<asr_text>URL 文本"}
    )
    # The URL downloader talks to the same fake host; a missing path must not
    # be treated as a successful download.
    proc = qasr.run(
        "--server", source.url, "transcribe",
        f"{source.url}/audio.wav", "--json", timeout=30,
    )
    assert proc.returncode in {3, 4}, proc.stderr
    assert "Traceback" not in proc.stderr
