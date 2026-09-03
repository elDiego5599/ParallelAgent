"""Self-healing: un reintento ante diff corrupto, luego salida limpia."""

import subprocess

import pytest

from conftest import init_git_repo
from git_bridge import GitBridgeError, apply_consensus_to_branch
from orchestrator import build_repair_prompt, finish_build_output
from providers import ProviderError


DIFF = "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1,2 @@\n hola\n+linea nueva\n"


def consensus_branches(path):
    r = subprocess.run(
        ["git", "branch", "--list", "consensus/*"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    return r.stdout


def test_heals_on_second_try(tmp_path):
    init_git_repo(tmp_path)
    seen = []
    rc = finish_build_output(
        tmp_path, "t", "basura no diff", context="ctx",
        repair_chat=lambda p: (seen.append(p), DIFF)[1],
    )
    assert rc == 0 and len(seen) == 1
    assert "falló" in seen[0] and "ctx" in seen[0]
    out = consensus_branches(tmp_path)
    assert "consensus/" in out
    branch = [l.strip().lstrip("* ") for l in out.splitlines() if "consensus/" in l][0]
    shown = subprocess.run(
        ["git", "show", f"{branch}:f.txt"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout
    assert shown == "hola\nlinea nueva\n"


def test_gives_up_after_failed_repair(tmp_path):
    init_git_repo(tmp_path)
    base = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    rc = finish_build_output(
        tmp_path, "t", "basura", repair_chat=lambda p: "más basura"
    )
    assert rc == 1
    assert "consensus/" not in consensus_branches(tmp_path)
    current = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert current == base


def test_no_repair_callback_keeps_old_behavior(tmp_path, capsys):
    init_git_repo(tmp_path)
    assert finish_build_output(tmp_path, "t", "basura") == 1
    assert "ERROR GIT" in capsys.readouterr().out


def test_repair_provider_error_is_clean(tmp_path):
    init_git_repo(tmp_path)

    def boom(prompt):
        raise ProviderError("caído")

    assert finish_build_output(tmp_path, "t", "basura", repair_chat=boom) == 1


def test_stderr_exposed_and_prompt_complete(tmp_path):
    init_git_repo(tmp_path)
    with pytest.raises(GitBridgeError) as exc:
        apply_consensus_to_branch(tmp_path, "t", "basura")
    assert exc.value.stderr
    prompt = build_repair_prompt("tarea X", "ERR123", "ctx-original")
    assert "ERR123" in prompt and "tarea X" in prompt
    assert "ctx-original" in prompt and "EXCLUSIVAMENTE" in prompt
