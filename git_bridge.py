"""Puente con Git para ParallelAgent.

Ciclo clásico (`apply_consensus_to_branch`): rama efímera transaccional,
si el diff falla vuelve al origen y borra la temporal.
Macro-cascada (`begin_consensus_branch` + `commit_diff_to_branch`): una
misma rama acumula N commits, un fallo en el ciclo N preserva los
commits 1..N-1. Solo modo build. Nunca escribe sobre la rama en curso
y nunca hace push salvo `push_branch` explícito (--push).
"""

from datetime import datetime
from pathlib import Path
import re
import subprocess


class GitBridgeError(Exception):
    """Fallo en validación, creación de rama o aplicación del diff."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


_FENCE_RE = re.compile(r"```(?:diff)?\s*\n(.*?)```", re.DOTALL)


def _extract_diff(diff_str: str) -> str:
    """Extrae el diff si el modelo lo envolvió en cercas Markdown.

    Garantiza un único salto final: git apply rechaza parches sin newline.
    """
    m = _FENCE_RE.search(diff_str or "")
    clean = m.group(1).strip() if m else (diff_str or "").strip()
    return clean + "\n" if clean else ""


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
        err = result.stderr.strip() or result.stdout.strip()
        raise GitBridgeError(f"git {' '.join(args)} falló: {err}", stderr=err)
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


def check_clean_working_tree(path: Path) -> None:
    """Aborta si hay cambios sin commitear (incluye untracked por --porcelain).

    Debe llamarse antes de cualquier `checkout -b`: con árbol sucio Git
    arrastraría el WIP a la rama temporal y `git add -A` lo commitearía
    en `consensus/...`, contaminando el diff y rompiendo la confianza.
    """
    status = _run_git(path, "status", "--porcelain")
    if status.strip():
        raise GitBridgeError(
            "El árbol de trabajo tiene cambios sin commitear (dirty working tree).\n"
            "Haz commit o stash de tus cambios antes de ejecutar 'build' "
            "para evitar contaminar la rama consensus/..."
        )


def _apply_checked(path: Path, diff_str: str) -> None:
    """Aplica con --recount (corrige hunks mal contados); fallback sin él.

    `--recount` rompe parches multi-archivo sin cabeceras `diff --git`
    (los confunde con contexto), mientras que el apply plano falla ante
    conteos tipo `@@ -1,9`. Probar ambos cubre los dos estilos de LLM.
    """
    try:
        _run_git(path, "apply", "--recount", "--check", input_text=diff_str)
        _run_git(path, "apply", "--recount", input_text=diff_str)
        return
    except GitBridgeError as recount_err:
        try:
            _run_git(path, "apply", "--check", input_text=diff_str)
            _run_git(path, "apply", input_text=diff_str)
            return
        except GitBridgeError:
            # Prioriza el error con --recount (más informativo para self-healing)
            raise recount_err


def diff_summary(diff_str: str) -> str:
    """Resumen humano del diff para pedir consentimiento antes de commitear.

    No toca Git: parsea cabeceras `diff --git`/`--- a/` y cuenta +/-
    (excluye cabeceras). Ej: '2 archivos, +12/-3: lib/a.dart, ...'.
    """
    clean = _extract_diff(diff_str)
    if not clean.strip():
        return "diff vacío"
    files: list = []
    added = removed = 0
    for line in clean.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" b/")
            if len(parts) == 2:
                f = parts[1].strip()
                if f not in files:
                    files.append(f)
        elif line.startswith("--- a/"):
            f = line[6:].strip()
            if f not in files:
                files.append(f)
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if not files:
        return f"+{added}/-{removed} (archivos sin cabecera reconocible)"
    shown = ", ".join(files[:6])
    extra = f" +{len(files) - 6} más" if len(files) > 6 else ""
    return f"{len(files)} archivo(s), +{added}/-{removed}: {shown}{extra}"


def begin_consensus_branch(path: Path, task: str) -> dict:
    """Crea la rama acumulativa de la macro-cascada (sin commitear nada).

    Exige árbol limpio. Vuelve a la rama original. El branch queda
    apuntando al HEAD actual; los ciclos harán commits encima.
    """
    if not path.is_dir():
        raise GitBridgeError(f"La ruta no es un directorio válido: {path}")
    if not is_git_repo(path):
        raise GitBridgeError(
            f"La ruta no es un repositorio Git: {path}. "
            "Inicialízalo con git init o apunta --path a uno válido."
        )
    check_clean_working_tree(path)
    original = current_branch(path)
    branch = build_branch_name(task)
    try:
        _run_git(path, "checkout", "-b", branch)
    except GitBridgeError:
        raise
    try:
        _run_git(path, "checkout", original)
    except GitBridgeError as e:
        print(f"[AVISO] No se pudo volver a '{original}': {e}. Quedaste en '{branch}'.")
    return {"original_branch": original, "branch": branch}


def commit_diff_to_branch(
    path: Path,
    branch: str,
    original: str,
    task: str,
    diff_str: str,
    cycle: int = 1,
    commit_message: str | None = None,
) -> dict:
    """Aplica un diff y commitea sobre la rama acumulativa existente.

    A diferencia del ciclo clásico, ante fallo NO borra la rama: preserva
    los commits de ciclos previos y vuelve a `original`. `original` es la
    rama previa a la macro (no cambia entre ciclos).
    """
    diff_str = _extract_diff(diff_str)
    if not diff_str:
        raise GitBridgeError("Diff vacío: nada que aplicar.")
    try:
        _run_git(path, "checkout", branch)
        _apply_checked(path, diff_str)
        _run_git(path, "add", "-A")
        _run_git(
            path, "commit", "-m",
            commit_message or f"consensus(c{cycle}): {task[:72]}",
        )
    except GitBridgeError:
        try:
            _run_git(path, "checkout", original)
        except GitBridgeError:
            pass
        raise
    try:
        _run_git(path, "checkout", original)
    except GitBridgeError as e:
        print(f"[AVISO] No se pudo volver a '{original}': {e}. Quedaste en '{branch}'.")
    return {"branch": branch, "cycle": cycle}


def push_branch(path: Path, branch: str, remote: str = "origin") -> str:
    """Push explícito de la rama consensus. Solo bajo --push del usuario.

    El texto de la tarea ('pushea esto') NUNCA dispara push: solo este
    llamado con flag explícito. Devuelve la salida de git push.
    """
    return _run_git(path, "push", "-u", remote, branch)


def delete_branch(path: Path, branch: str) -> None:
    """Borra una rama acumulativa vacía (0 commits por declinar usuario).

    Nunca lanza: es limpieza best-effort.
    """
    try:
        _run_git(path, "branch", "-D", branch)
    except GitBridgeError:
        pass


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
    """Valida el repo, exige árbol limpio, crea rama efímera, aplica y commitea.

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
    diff_str = _extract_diff(diff_str)
    if not diff_str:
        raise GitBridgeError("Diff vacío: nada que aplicar.")

    # Primero, no hacer daño: abortar antes de cualquier checkout -b.
    check_clean_working_tree(path)

    dirty = False
    original = current_branch(path)
    branch = build_branch_name(task)
    created = False

    try:
        _run_git(path, "checkout", "-b", branch)
        created = True
        _apply_checked(path, diff_str)
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

    try:
        _run_git(path, "checkout", original)
    except GitBridgeError as e:
        print(f"[AVISO] No se pudo volver a '{original}': {e}. Quedaste en '{branch}'.")

    return {
        "original_branch": original,
        "branch": branch,
        "dirty_before": dirty,
        "inspect": f"git diff {original}...{branch}",
    }
