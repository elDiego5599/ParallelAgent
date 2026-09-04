"""Puente Git: rama efímera, rollback y sanidad."""

import re
import subprocess

import pytest

from git_bridge import (
    GitBridgeError,
    _extract_diff,
    apply_consensus_to_branch,
    build_branch_name,
    check_clean_working_tree,
    current_branch,
    is_dirty,
    is_git_repo,
    slugify_task,
)


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def init_repo(path):
    git("init", "-q", cwd=path)
    git("config", "user.email", "t@t.t", cwd=path)
    git("config", "user.name", "t", cwd=path)
    (path / "f.txt").write_text("hola\n")
    git("add", "-A", cwd=path)
    git("-c", "user.name=t", "-c", "user.email=t@t.t", "commit", "-qm", "init", cwd=path)


DIFF = """--- a/f.txt
+++ b/f.txt
@@ -1 +1,2 @@
 hola
+linea nueva
"""


def show_file(path, branch, name="f.txt"):
    r = subprocess.run(
        ["git", "show", f"{branch}:{name}"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    return r.stdout


def test_happy_path_creates_branch_and_applies(tmp_path):
    init_repo(tmp_path)
    base = current_branch(tmp_path)
    info = apply_consensus_to_branch(tmp_path, "Agrega línea", DIFF)
    assert info["original_branch"] == base
    assert re.fullmatch(r"consensus/\d{8}-\d{6}-[\w-]+", info["branch"])
    assert show_file(tmp_path, info["branch"]) == "hola\nlinea nueva\n"
    assert current_branch(tmp_path) == base
    assert info["inspect"] == f"git diff {base}...{info['branch']}"
    assert info["dirty_before"] is False


def test_corrupt_diff_rolls_back_without_trash(tmp_path):
    init_repo(tmp_path)
    base = current_branch(tmp_path)
    with pytest.raises(GitBridgeError):
        apply_consensus_to_branch(tmp_path, "roto", "esto no es un diff")
    assert current_branch(tmp_path) == base
    branches = subprocess.run(
        ["git", "branch", "--list", "consensus/*"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout
    assert "consensus/" not in branches


def test_rejects_non_repo_and_empty_diff(tmp_path):
    with pytest.raises(GitBridgeError, match="no es un repositorio Git"):
        apply_consensus_to_branch(tmp_path, "t", DIFF)
    init_repo(tmp_path)
    with pytest.raises(GitBridgeError, match="Diff vacío"):
        apply_consensus_to_branch(tmp_path, "t", "   ")


def test_dirty_is_reported_not_blocked(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("cambio sin commitear\n")
    assert is_dirty(tmp_path) is True
    assert is_git_repo(tmp_path) is True


def _consensus_branches(path):
    return subprocess.run(
        ["git", "branch", "--list", "consensus/*"],
        cwd=path, capture_output=True, text=True, check=True,
    ).stdout


def test_dirty_tracked_aborts_without_branch(tmp_path):
    init_repo(tmp_path)
    base = current_branch(tmp_path)
    (tmp_path / "f.txt").write_text("WIP del usuario\n")
    with pytest.raises(GitBridgeError, match="sin commitear"):
        apply_consensus_to_branch(tmp_path, "tarea x", DIFF)
    assert "consensus/" not in _consensus_branches(tmp_path)
    assert current_branch(tmp_path) == base
    # WIP intacto, no fue commiteado ni arrastrado
    assert (tmp_path / "f.txt").read_text() == "WIP del usuario\n"


def test_dirty_untracked_aborts_without_branch(tmp_path):
    init_repo(tmp_path)
    base = current_branch(tmp_path)
    (tmp_path / "nuevo.txt").write_text("wip untracked\n")
    with pytest.raises(GitBridgeError, match="sin commitear"):
        apply_consensus_to_branch(tmp_path, "tarea x", DIFF)
    assert "consensus/" not in _consensus_branches(tmp_path)
    assert current_branch(tmp_path) == base
    assert (tmp_path / "nuevo.txt").read_text() == "wip untracked\n"


def test_check_clean_working_tree(tmp_path):
    init_repo(tmp_path)
    check_clean_working_tree(tmp_path)  # limpio: no lanza
    (tmp_path / "f.txt").write_text("sucio\n")
    with pytest.raises(GitBridgeError, match="stash"):
        check_clean_working_tree(tmp_path)


def test_finish_build_output_dirty_no_repair(tmp_path):
    from orchestrator import finish_build_output

    init_repo(tmp_path)
    (tmp_path / "f.txt").write_text("WIP\n")
    called = []

    def boom(prompt):
        called.append(prompt)
        return DIFF

    rc = finish_build_output(tmp_path, "t", DIFF, repair_chat=boom)
    assert rc == 1
    assert called == []  # fatal: no debe intentar auto-reparación
    assert "consensus/" not in _consensus_branches(tmp_path)


def test_slug_and_branch_name():
    assert slugify_task("Agrega logs y try/catch!!!") == "agrega-logs-y-trycatch"
    assert build_branch_name("t").startswith("consensus/")


def test_extract_diff_strips_fences():
    assert _extract_diff("```diff\n--- a/f\n+++ b/f\n```") == "--- a/f\n+++ b/f\n"
    assert _extract_diff("texto\n```\nDIFF\n```\ncola") == "DIFF\n"
    assert _extract_diff("  DIFF crudo  ") == "DIFF crudo\n"
    assert _extract_diff("```\n```") == ""
    assert _extract_diff("   ") == ""


def test_recount_saves_miscounted_hunk(tmp_path):
    init_repo(tmp_path)
    bad_counts = """--- a/f.txt
+++ b/f.txt
@@ -1,9 +1,9 @@
 hola
+linea nueva
"""
    info = apply_consensus_to_branch(tmp_path, "t", bad_counts)
    assert show_file(tmp_path, info["branch"]) == "hola\nlinea nueva\n"
    assert info["branch"].startswith("consensus/")
