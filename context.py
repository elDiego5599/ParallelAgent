"""Constructor de mapa de contexto para repositorios en ParallelAgent.

Extrae la estructura del proyecto y los fragmentos de código más relevantes
según la tarea técnica, respetando .gitignore y presupuestos estrictos de caracteres.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import List, Set


# Extensiones de código habituales en proyectos polyglot (Flutter/C++/Java/TS/Python, etc.)
VALID_EXTENSIONS: Set[str] = {
    # C/C++
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hxx",
    # Dart / Flutter
    ".dart",
    ".yaml",
    # JVM
    ".java",
    ".kt",
    ".kts",
    ".gradle",
    # Web / Scripting
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".py",
    # Configuración de compilación
    ".cmake",
    ".json",
    ".xml",
    ".toml",
    ".md",
    "CMakeLists.txt",
}


# Archivos o directorios que jamás deben leerse aunque git los rastree
EXCLUDED_PATTERNS: Set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "pubspec.lock",
    "Cargo.lock",
    "poetry.lock",
}


@dataclass
class ScoredFile:
    path: Path
    rel_path: str
    score: float
    content: str = ""


def is_binary_file(file_path: Path) -> bool:
    """Verifica de forma rápida si un archivo es binario leyendo sus primeros 1024 bytes."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
        return False
    except OSError:
        return True


def get_repo_files(repo_path: Path) -> List[Path]:
    """Obtiene la lista de archivos rastreados y no rastreados que NO estén en .gitignore

    usando git ls-files. Si falla o no es repo git, recurre a rglob.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in result.stdout.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue
            full_path = repo_path / clean_line
            if full_path.is_file():
                files.append(full_path)
        return files
    except (subprocess.SubprocessError, FileNotFoundError):
        # Fallback si no está inicializado git aún
        return [
            p
            for p in repo_path.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ]


def extract_keywords(task: str) -> Set[str]:
    """Extrae palabras clave significativas de la descripción de la tarea."""
    # Extrae tokens alfanuméricos de al menos 3 caracteres
    words = re.findall(r"[A-Za-z0-9_]{3,}", task.lower())
    # Stopwords técnicas básicas para no sesgar por palabras comunes
    stopwords = {
        "para",
        "como",
        "este",
        "esta",
        "todo",
        "hacer",
        "fix",
        "bug",
        "que",
        "los",
        "las",
        "del",
        "con",
        "por",
    }
    return {w for w in words if w not in stopwords}


def score_file(rel_path: str, content: str, keywords: Set[str]) -> float:
    """Calcula la relevancia de un archivo según la presencia de palabras clave

    tanto en la ruta como en el contenido.
    """
    score = 0.0
    rel_lower = rel_path.lower()
    content_lower = content.lower()

    for kw in keywords:
        # Match directo en el nombre del archivo o ruta tiene un peso alto
        if kw in rel_lower:
            score += 15.0

        # Ocurrencias en el contenido del archivo
        occurrences = content_lower.count(kw)
        if occurrences > 0:
            # Puntuación logarítmica/acotada para no premiar archivos gigantes
            score += min(occurrences * 1.5, 10.0)

    # Bonificación para extensiones nativas o críticas si se mencionan en las keywords
    if any(k in ["c++", "cpp", "native", "leak", "bridge", "jni"] for k in keywords):
        if rel_lower.endswith((".cpp", ".h", ".hpp", ".cc")):
            score += 5.0

    return score


def build_repo_tree(files: List[Path], repo_path: Path, max_lines: int = 40) -> str:
    """Genera una vista previa textual en árbol del repositorio."""
    rel_paths = sorted([str(f.relative_to(repo_path)) for f in files])
    if len(rel_paths) > max_lines:
        preview = rel_paths[:max_lines]
        preview.append(f"... y {len(rel_paths) - max_lines} archivos más.")
        return "\n".join(f"  - {p}" for p in preview)
    return "\n".join(f"  - {p}" for p in rel_paths)


def build_repo_context(
    repo_path: Path,
    task: str,
    max_total_chars: int = 12000,
    max_file_chars: int = 4000,
) -> str:
    """Construye el mapa de contexto consolidado para inyectar en el prompt inicial.

    Args:
        repo_path: Ruta raíz del proyecto.
        task: Descripción técnica del requerimiento.
        max_total_chars: Presupuesto total de caracteres de contexto (~3k tokens,
            seguro bajo límites TPM de 8000 de tiers gratuitos).
        max_file_chars: Límite por archivo individual para evitar monopolios.

    Returns:
        Cadena formateada con el mapa del repo y los archivos más relevantes.
    """
    all_files = get_repo_files(repo_path)
    keywords = extract_keywords(task)

    scored_files: List[ScoredFile] = []

    for file_path in all_files:
        # Filtrado de exclusiones y binarios
        if file_path.name in EXCLUDED_PATTERNS:
            continue
        if (
            file_path.suffix.lower() not in VALID_EXTENSIONS
            and file_path.name != "CMakeLists.txt"
        ):
            continue
        if is_binary_file(file_path):
            continue

        try:
            rel_path = str(file_path.relative_to(repo_path))
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        score = score_file(rel_path, content, keywords)
        scored_files.append(
            ScoredFile(
                path=file_path, rel_path=rel_path, score=score, content=content
            )
        )

    # Ordenar por relevancia descendente
    scored_files.sort(key=lambda x: x.score, reverse=True)

    # 1. Armar el árbol del proyecto
    tree_view = build_repo_tree(all_files, repo_path)

    header = (
        f"=== MAPA DE CONTEXTO DEL REPOSITORIO ===\n"
        f"Estructura general del proyecto:\n{tree_view}\n\n"
        f"Archivos relevantes seleccionados para la tarea:\n"
    )

    current_chars = len(header)
    included_blocks = []

    for item in scored_files:
        # Si ya se agotó el presupuesto y al menos metimos el árbol, paramos
        if current_chars >= max_total_chars:
            break

        # Truncado por archivo si es excesivamente largo
        content = item.content
        if len(content) > max_file_chars:
            content = (
                content[:max_file_chars]
                + f"\n\n[... Archivo truncado por longitud. Tamaño original: {len(item.content)} caracteres ...]"
            )

        block = (
            f"--- [ARCHIVO: {item.rel_path}] (Relevancia: {item.score:.1f}) ---\n"
            f"```\n{content}\n```\n\n"
        )

        if current_chars + len(block) > max_total_chars:
            # Si no cabe completo, tomamos lo que quepa descontando el envoltorio
            remaining = max_total_chars - current_chars
            if remaining > 1000:
                prefix = f"--- [ARCHIVO: {item.rel_path}] (Parcial) ---\n```\n"
                suffix = "\n[... Truncado por límite de contexto ...]\n```\n\n"
                room = remaining - len(prefix) - len(suffix)
                if room > 0:
                    included_blocks.append(prefix + content[:room] + suffix)
            break

        included_blocks.append(block)
        current_chars += len(block)

    return header + "".join(included_blocks)
