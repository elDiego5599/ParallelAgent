"""Puente con Git para ParalelAgent.

Aplica el parche acordado sobre una rama efímera de forma transaccional:
si el diff falla, vuelve a la rama original y borra la temporal.
Solo se usa en modo build. Nunca escribe sobre la rama en curso.
"""

from datetime import datetime
from pathlib import Path
import re
import subprocess


class GitBridgeError(Exception):
    """Fallo en validación, creación de rama o aplicación del diff."""

    pass


def _run_git(path: Path, *args: str, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=path,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError) as e:
        raise GitBridgeError(f"Fallo al ejecutar git {' '.join(args)}: {e}")
    if result.returncode != 0:
        raise GitBridgeError(
            f"git {' '.join(args)} falló: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def is_git_repo(path: Path) -> bool:
    try:
        out = _run_git(path, "rev-parse", "--is-inside-work-tree")
        return out == "true"
    except GitBridgeError:
        return False


def is_dirty(path: Path) -> bool:
    return bool(_run_git(path, "status", "--porcelain"))


def current_branch(path: Path) -> str:
    return _run_git(path, "rev-parse", "--abbrev-ref", "HEAD")


def slugify_task(task: str, max_words: int = 5) -> str:
    words = re.sub(r"[^a-zA-Z0-9áéíóúñü ]", "", task.lower()).split()
    slug = "-".join(words[:max_words]) or "tarea"
    return slug[:50]


def build_branch_name(task: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"consensus/{stamp}-{slugify_task(task)}"


def apply_consensus_to_branch(
    path: Path,
    task: str,
    diff_str: str,
    commit_message: str | None = None,
) -> dict:
    """Valida el repo, crea rama efímera, aplica el diff y hace commit.

    Devuelve dict con rama original, rama creada e instrucciones de revisión.
    Ante cualquier fallo, restaura la rama original y borra la temporal.
    """
    if not path.is_dir():
        raise GitBridgeError(f"La ruta no es un directorio válido: {path}")
    if not is_git_repo(path):
        raise GitBridgeError(
            f"La ruta no es un repositorio Git: {path}. "
            "Inicialízalo con git init o apunta --path a uno válido."
        )
    if not diff_str or not diff_str.strip():
        raise GitBridgeError("Diff vacío: nada que aplicar.")

    dirty = is_dirty(path)
    original = current_branch(path)
    branch = build_branch_name(task)
    created = False

    try:
        _run_git(path, "checkout", "-b", branch)
        created = True
        _run_git(path, "apply", "--check", input_text=diff_str)
        _run_git(path, "apply", input_text=diff_str)
        _run_git(path, "add", "-A")
        _run_git(
            path, "commit", "-m", commit_message or f"consensus: {task[:72]}"
        )
    except GitBridgeError:
        _run_git(path, "checkout", original)
        if created:
            try:
                _run_git(path, "branch", "-D", branch)
            except GitBridgeError:
                pass
        raise

    return {
        "original_branch": original,
        "branch": branch,
        "dirty_before": dirty,
        "inspect": f"git diff {original}...{branch}",
    }
