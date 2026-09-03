"""Puente Git: rama efímera, rollback y sanidad."""

import re
import subprocess

import pytest

from git_bridge import (
    GitBridgeError,
    _extract_diff,
    apply_consensus_to_branch,
    build_branch_name,
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


def test_happy_path_creates_branch_and_applies(tmp_path):
    init_repo(tmp_path)
    base = current_branch(tmp_path)
    info = apply_consensus_to_branch(tmp_path, "Agrega línea", DIFF)
    assert info["original_branch"] == base
    assert re.fullmatch(r"consensus/\d{8}-\d{6}-[\w-]+", info["branch"])
    assert (tmp_path / "f.txt").read_text() == "hola\nlinea nueva\n"
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
    assert (tmp_path / "f.txt").read_text() == "hola\nlinea nueva\n"
    assert info["branch"].startswith("consensus/")
