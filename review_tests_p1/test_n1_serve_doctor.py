"""N1: serve thin wrapper and doctor --server."""

from __future__ import annotations

import json
import os

from fake_openai import Response, error_response, json_response
from helpers import assert_exit, find_check, parse_json, scan_text_for_secret


def test_serve_dry_run_json(qasr):
    proc = qasr.run("serve", "--dry-run", "--json")
    assert_exit(proc, 0, context="serve dry-run")
    payload = parse_json(proc)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["command"] == "serve"
    argv = payload.get("argv")
    assert isinstance(argv, list) and argv
    assert payload.get("binary")
    assert payload.get("model")
    assert payload.get("resolved")
    assert payload.get("source") in {"local", "repo"}


def test_serve_known_and_unknown_args_passthrough(qasr):
    proc = qasr.run(
        "serve", "0.6b",
        "--host", "0.0.0.0", "--port", "18000",
        "--gpu-memory-utilization", "0.8",
        "--tensor-parallel-size", "2",
        "--foo=bar",
        "--dry-run", "--json",
    )
    assert_exit(proc, 0, context="serve passthrough")
    payload = parse_json(proc)
    argv = [str(item) for item in payload["argv"]]
    joined = " ".join(argv)
    assert "0.0.0.0" in joined
    assert "18000" in joined
    assert "0.8" in joined
    assert "--tensor-parallel-size" in argv
    assert "2" in argv
    assert "--foo=bar" in argv


def test_serve_missing_binary_exits_3(qasr, empty_path):
    proc = qasr.run(
        "serve", "0.6b",
        env={"PATH": str(empty_path)},
        timeout=30,
    )
    assert_exit(proc, 3, context="serve missing binary")
    assert "qwen-asr-serve" in (proc.stdout + proc.stderr)
    assert "Traceback" not in proc.stderr


def test_serve_execs_fake_binary_with_argv(qasr, fake_serve_bin):
    bin_dir, argv_file = fake_serve_bin
    proc = qasr.run(
        "serve", "0.6b",
        "--host", "127.0.0.1", "--port", "18000",
        "--tensor-parallel-size", "2",
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "QASR_TEST_ARGV_FILE": str(argv_file),
        },
        timeout=30,
    )
    assert_exit(proc, 0, context="serve fake exec")
    assert proc.stdout.strip() == "", proc.stdout
    assert argv_file.is_file(), "fake qwen-asr-serve was not executed"
    argv = argv_file.read_text(encoding="utf-8").splitlines()
    joined = " ".join(argv)
    assert "0.6B" in joined or "0.6b" in joined
    assert "18000" in argv
    assert "--tensor-parallel-size" in argv


def test_serve_rejects_gpu_memory_utilization_out_of_range(qasr):
    proc = qasr.run(
        "serve", "--gpu-memory-utilization", "2",
        "--dry-run", "--json",
    )
    assert_exit(proc, 2, context="serve gpu-memory-utilization range")


def test_serve_model_after_unknown_flag_is_still_the_model(qasr):
    proc = qasr.run(
        "serve", "--tensor-parallel-size", "2", "0.6b",
        "--dry-run", "--json",
    )
    assert_exit(proc, 0, context="serve model after unknown flag")
    payload = parse_json(proc)
    assert payload["model"] == "0.6b", payload
    assert payload["resolved"] in {"0.6b", "Qwen/Qwen3-ASR-0.6B"} or \
        str(payload["resolved"]).endswith("Qwen3-ASR-0.6B"), payload


def test_serve_dry_run_has_declared_schema(qasr, repo_root):
    import jsonschema
    from helpers import load_schema

    proc = qasr.run("serve", "0.6b", "--dry-run", "--json")
    assert_exit(proc, 0, context="serve schema")
    payload = parse_json(proc)
    if (repo_root / "schemas" / "serve.json").is_file():
        jsonschema.validate(payload, load_schema("serve.json"))
        return
    schema = load_schema("dry-run.json")
    assert "serve" in schema["properties"]["command"].get("enum", []), schema
    jsonschema.validate(payload, schema)


def test_serve_dry_run_does_not_import_runtime(qasr, import_blocker, repo_root):
    blocker, marker = import_blocker
    proc = qasr.run(
        "serve", "--dry-run", "--json",
        env={"PYTHONPATH": f"{blocker}{os.pathsep}{repo_root}"},
    )
    assert_exit(proc, 0, context="serve dry-run import blocker")
    assert not marker.exists(), "serve --dry-run imported torch/qwen_asr/vllm"


def test_doctor_server_models_ok(qasr, fake_openai):
    server = fake_openai(models=["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"])
    proc = qasr.run("doctor", "--server", server.url, "--json")
    assert_exit(proc, 0, context="doctor --server")
    payload = parse_json(proc)
    check = find_check(payload, "server")
    assert check is not None, f"no server check in {payload}"
    assert check["status"] == "ok", check
    detail = json.dumps(check, ensure_ascii=False)
    assert "Qwen/Qwen3-ASR" in detail


def test_server_precedence_cli_over_env_over_home_config(qasr):
    (qasr.home / "config.json").write_text(
        json.dumps({"server": "http://home-config:1"}), encoding="utf-8"
    )
    env = {"QASR_SERVE_URL": "http://env-config:2"}

    from_home = qasr.run("config", "show", "--json", env=env)
    assert_exit(from_home, 0, context="server precedence env")
    env_value = parse_json(from_home)["config"]["server"]
    assert env_value["value"] == "http://env-config:2", env_value
    assert env_value["source"] == "env:QASR_SERVE_URL", env_value

    from_cli = qasr.run(
        "--server", "http://cli-config:3", "config", "show", "--json", env=env,
    )
    assert_exit(from_cli, 0, context="server precedence cli")
    cli_value = parse_json(from_cli)["config"]["server"]
    assert cli_value["value"] == "http://cli-config:3", cli_value
    assert cli_value["source"] == "cli", cli_value


def test_doctor_server_401_is_reported_without_failing_default(qasr, fake_openai):
    server = fake_openai(auth_token="secret-token", require_auth=True)
    proc = qasr.run("doctor", "--server", server.url, "--json")
    assert_exit(proc, 0, context="doctor 401 default")
    payload = parse_json(proc)
    check = find_check(payload, "server")
    assert check is not None, payload
    assert check["status"] in {"warn", "fail"}, check

    strict = qasr.run("doctor", "--server", server.url, "--json", "--strict")
    assert_exit(strict, 3, context="doctor 401 strict")


def test_doctor_server_401_exposes_machine_readable_status(qasr, fake_openai):
    server = fake_openai(auth_token="{{SECRET_e5r8t1uv}}", require_auth=True)
    proc = qasr.run(
        "doctor", "--server", server.url, "--json",
        env={"QASR_SERVE_TOKEN": "{{SECRET_w9y4n2cb}}"},
    )
    assert_exit(proc, 0, context="doctor 401 machine-readable")
    check = find_check(parse_json(proc), "server")
    assert check is not None, proc.stdout
    assert check["value"].get("http_status") == 401, check
    assert check["value"].get("auth_rejected") is True, check


def test_doctor_server_token_not_in_output(qasr, fake_openai):
    token = "sentinel-token-6wyb4joe"
    server = fake_openai(auth_token=token, require_auth=True)
    proc = qasr.run(
        "doctor", "--server", server.url, "--json",
        env={"QASR_SERVE_TOKEN": token},
    )
    assert_exit(proc, 0, context="doctor token")
    assert token not in proc.stdout
    assert token not in proc.stderr


def test_doctor_server_500_is_reported(qasr, fake_openai):
    server = fake_openai(
        models_response=lambda request, srv: error_response("server exploded", status=500)
    )
    proc = qasr.run("doctor", "--server", server.url, "--json")
    assert_exit(proc, 0, context="doctor 500")
    payload = parse_json(proc)
    assert find_check(payload, "server") is not None


def test_doctor_server_timeout_is_reported(qasr, fake_openai):
    server = fake_openai(
        models_response=lambda request, srv: json_response(
            {"object": "list", "data": []}, delay=6.0
        )
    )
    proc = qasr.run(
        "doctor", "--server", server.url, "--json",
        timeout=25,
    )
    assert_exit(proc, 0, context="doctor timeout")
    payload = parse_json(proc)
    check = find_check(payload, "server")
    assert check is not None, payload
    assert check["status"] in {"warn", "fail"}, check
    detail = json.dumps(check).lower()
    assert "timeout" in detail or "timed out" in detail or "超时" in detail
    assert "Traceback" not in proc.stderr


def test_doctor_server_invalid_url_does_not_crash(qasr):
    proc = qasr.run("doctor", "--server", "not-a-url", "--json", timeout=20)
    assert proc.returncode in (0, 2), proc.stderr
    assert "Traceback" not in proc.stderr


def test_doctor_server_userinfo_is_redacted(qasr, fake_openai):
    secret = "{{SECRET_n4v8c2we}}"
    server = fake_openai(models=["Qwen/Qwen3-ASR-0.6B"])
    url = server.url.replace("http://", f"http://user:{secret}@", 1)
    proc = qasr.run("doctor", "--server", url, "--json", timeout=20)
    assert_exit(proc, 0, context="doctor userinfo")
    payload = parse_json(proc)
    check = find_check(payload, "server")
    assert check is not None, payload
    assert check["status"] == "ok", check
    assert check["value"]["models"] == ["Qwen/Qwen3-ASR-0.6B"], check
    assert check["value"]["url"] == server.url
    assert not scan_text_for_secret(proc.stdout, secret)
    assert not scan_text_for_secret(proc.stderr, secret)


def test_doctor_invalid_port_with_userinfo_is_classified_not_crashed(qasr):
    secret = "{{SECRET_k7m3x9pd}}"
    url = f"http://user:{secret}@127.0.0.1:99999"
    proc = qasr.run("doctor", "--server", url, "--json", timeout=20)
    assert_exit(proc, 0, context="doctor invalid port")
    payload = parse_json(proc)
    assert find_check(payload, "server") is not None, payload
    assert not scan_text_for_secret(proc.stdout, secret)


def test_doctor_same_host_other_port_redirect_does_not_forward_token(
    qasr, fake_openai
):
    token = "{{SECRET_q2w6e8rt}}"
    target = fake_openai(auth_token=token, require_auth=True)
    source = fake_openai(
        models_response=lambda request, srv: Response(
            status=302,
            headers={"Location": f"{target.url}/v1/models"},
        )
    )
    proc = qasr.run(
        "doctor", "--server", source.url, "--json",
        env={"QASR_SERVE_TOKEN": token}, timeout=20,
    )
    assert_exit(proc, 0, context="doctor cross-port redirect")
    payload = parse_json(proc)
    check = find_check(payload, "server")
    assert check is not None, payload
    assert check["status"] in {"warn", "fail"}, check
    assert not [r for r in target.requests if r.authorization], target.requests
    assert not scan_text_for_secret(proc.stdout, token)
    assert not scan_text_for_secret(proc.stderr, token)


def test_doctor_same_origin_redirect_chain_is_capped(qasr, fake_openai):
    def models_response(request, srv):
        step = int((request.query.get("n") or ["0"])[0])
        if step < 10:
            return Response(
                status=302,
                headers={"Location": f"/v1/models?n={step + 1}"},
            )
        return json_response({"object": "list", "data": []})

    server = fake_openai(models_response=models_response)
    proc = qasr.run("doctor", "--server", server.url, "--json", timeout=20)
    assert_exit(proc, 0, context="doctor redirect cap")
    payload = parse_json(proc)
    check = find_check(payload, "server")
    assert check is not None, payload
    assert len(server.requests) <= 4, [r.path for r in server.requests]

