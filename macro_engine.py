"""Motor de Cascada Condicional (Adaptive Cascade Engine).

Fast-exit por defecto: 1 debate + 1 micro-auditoría barata. Solo se expande
a más ciclos si la auditoría canta NUEVO_HALLAZGO. Sin commits ni push
silenciosos: cada commit local pide consentimiento salvo --yes o modo no
interactivo (la invocación ya es consentimiento local); el push solo ocurre
con --push explícito (el texto de la tarea nunca lo dispara).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable, List, Optional

from git_bridge import (
    GitBridgeError,
    begin_consensus_branch,
    commit_diff_to_branch,
    diff_summary,
    push_branch,
)
from orchestrator import build_repair_prompt
from providers import ProviderError


AUDIT_DONE = "TAREA_FINALIZADA"
AUDIT_FINDING = "NUEVO_HALLAZGO"

_AUDIT_RE = re.compile(
    r"[*_]{0,2}\s*ESTADO:\s*(TAREA_FINALIZADA|NUEVO_HALLAZGO)\s*:?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_audit_estado(text: str) -> tuple:
    """Extrae (estado, hallazgo). Sin marcador o malformado -> TAREA_FINALIZADA.

    Fast-exit seguro: solo un NUEVO_HALLAZGO explícito expande la cascada.
    """
    matches = _AUDIT_RE.findall(text or "")
    if not matches:
        return AUDIT_DONE, ""
    estado, rest = matches[-1][0].upper(), (matches[-1][1] or "").strip()
    if estado != AUDIT_FINDING:
        return AUDIT_DONE, ""
    finding = re.sub(r"[*_`#>\-]*", "", rest).strip()
    finding = finding[:600] if len(finding) > 600 else finding
    return AUDIT_FINDING, finding


def build_audit_prompt(
    task: str,
    cycle: int,
    files_changed: List[str],
    summary: str,
    macro_history: str = "",
) -> str:
    files = ", ".join(files_changed) if files_changed else "(sin archivos declarados)"
    parts = [
        f"[Micro-auditoría tras ciclo {cycle}] Tarea vigente: {task}",
        f"Archivos del ciclo: {files}",
        f"Cambio aplicado: {summary}",
    ]
    if macro_history:
        parts.append(f"Historial macro:\n{macro_history}")
    parts.append(
        "Mirando el estado resultante del repositorio: ¿hay alguna regresión "
        "crítica o cabo suelto DIRECTO que impida dar por cerrada la tarea?\n"
        "Responde estrictamente con una línea: "
        "ESTADO: TAREA_FINALIZADA o ESTADO: NUEVO_HALLAZGO: <descripción concisa>. "
        "Nada de debates ni código, solo el estado."
    )
    return "\n\n".join(parts)


_STOPWORDS = {
    "para", "como", "este", "esta", "estos", "estas", "todo", "toda",
    "hacer", "hace", "hacen", "cada", "entre", "sobre", "desde",
    "donde", "cuando", "porque", "pero", "para", "with", "from",
    "that", "this", "have", "with", "para", "los", "las", "del",
    "con", "por", "una", "unos", "fallo", "error", "nuevo", "ciclo",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[A-Za-z0-9_]{4,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


@dataclass
class CycleBundle:
    result: object
    diff: str
    files: List[str] = field(default_factory=list)
    writer_label: str = ""
    repair_context: str = ""
    audit_chat: object = None
    repair_chat: object = None


class PingPongDetector:
    """Cortacircuitos anti-bucle oscilatorio entre ciclos."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.seen: List[dict] = []

    def observe(self, files: List[str], finding: str) -> None:
        self.seen.append({"files": set(files or []), "kw": _keywords(finding)})

    def is_oscillation(self, files: List[str], finding: str) -> tuple:
        new_files, new_kw = set(files or []), _keywords(finding)
        for prev in self.seen:
            if not prev["files"] and not new_files:
                continue
            file_overlap = bool(prev["files"] & new_files) if new_files else False
            both_empty_kw = not prev["kw"] and not new_kw
            if both_empty_kw:
                jaccard = 1.0 if file_overlap else 0.0
            elif not prev["kw"] or not new_kw:
                jaccard = 0.0
            else:
                inter = len(prev["kw"] & new_kw)
                union = len(prev["kw"] | new_kw)
                jaccard = inter / union if union else 0.0
            if file_overlap and jaccard >= self.threshold:
                return True, (
                    f"Solape en {sorted(prev['files'] & new_files)} "
                    f"con similitud {jaccard:.2f}: probable revert entre ciclos."
                )
            if new_files and prev["files"] == new_files and jaccard >= 0.4:
                return True, "Mismos archivos con hallazgo casi idéntico: bucle."
        return False, ""


def default_ask_commit(cycle: int, summary: str, branch: str, files: List[str]) -> bool:
    print("=" * 65)
    print(f"[MACRO] Ciclo {cycle} listo para commit en '{branch}'")
    print(f"  Cambio: {summary}")
    if files:
        print(f"  Archivos: {', '.join(files)}")
    try:
        ans = input("¿Commitear este ciclo? [S/n] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[MACRO] Entrada cancelada: se declina el commit sin traceback.")
        return False
    return ans in ("", "s", "si", "sí", "y", "yes")


def default_ask_continue(cycle: int, finding: str) -> bool:
    print(f"[MACRO] Hallazgo tras ciclo {cycle}: {finding}")
    try:
        ans = input(f"¿Abrir ciclo {cycle + 1} para este hallazgo? [S/n] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n[MACRO] Entrada cancelada: se frena la cascada sin traceback.")
        return False
    return ans in ("", "s", "si", "sí", "y", "yes")


@dataclass
class MacroResult:
    branch: str = ""
    original_branch: str = ""
    cycles_used: int = 0
    commits: int = 0
    findings: List[str] = field(default_factory=list)
    finished: bool = False
    stopped_reason: str = ""
    pushed: bool = False


class MacroEngine:
    """Envuelve un ciclo (peer o lead) con cascada condicional y consentimiento."""

    def __init__(
        self,
        project_path: Path,
        initial_task: str,
        max_cycles: int = 3,
        cycle_fn=None,
        interactive: bool = True,
        auto_approve: bool = False,
        push: bool = False,
        ask_commit=None,
        ask_continue=None,
        verbose: bool = True,
    ):
        self.project_path = project_path
        self.initial_task = initial_task
        self.max_cycles = max(1, max_cycles)
        self.cycle_fn = cycle_fn
        self.interactive = interactive
        self.auto_approve = auto_approve
        self.push = push
        self.ask_commit = ask_commit or default_ask_commit
        self.ask_continue = ask_continue or default_ask_continue
        self.verbose = verbose
        self.detector = PingPongDetector()

    def _consent_commit(self, cycle, summary, branch, files) -> bool:
        if self.auto_approve or not self.interactive:
            return True
        try:
            return bool(self.ask_commit(cycle, summary, branch, files))
        except (Exception, KeyboardInterrupt):
            return False

    def _consent_continue(self, cycle, finding) -> bool:
        if self.auto_approve or not self.interactive:
            return True
        try:
            return bool(self.ask_continue(cycle, finding))
        except (Exception, KeyboardInterrupt):
            return False

    def _audit(self, bundle: CycleBundle, task: str, cycle: int, history: str) -> tuple:
        summary = diff_summary(bundle.diff)
        prompt = build_audit_prompt(task, cycle, bundle.files, summary, history)
        try:
            response = bundle.audit_chat(prompt)
        except (ProviderError, TypeError, AttributeError):
            return AUDIT_DONE, ""
        except Exception:
            return AUDIT_DONE, ""
        return parse_audit_estado(response)

    def run(self) -> tuple:
        """Ejecuta la cascada. Devuelve (exit_code, MacroResult)."""
        if self.cycle_fn is None:
            raise ValueError("MacroEngine requiere cycle_fn(task) -> CycleBundle.")
        if self.max_cycles <= 1:
            bundle = self.cycle_fn(self.initial_task)
            return 0, MacroResult(cycles_used=1, finished=True, stopped_reason="single")
        try:
            info = begin_consensus_branch(self.project_path, self.initial_task)
        except GitBridgeError as e:
            print(f"[ERROR GIT] {e}")
            return 1, MacroResult(stopped_reason="dirty-or-git")
        original, branch = info["original_branch"], info["branch"]
        if self.verbose:
            print(f"[MACRO] Rama acumulativa: {branch} (origen: {original})")
        current_task = self.initial_task
        history_lines: List[str] = []
        findings: List[str] = []
        commits = 0
        cycles_used = 0
        finished = False
        reason = ""
        for cycle in range(1, self.max_cycles + 1):
            cycles_used = cycle
            bundle = self.cycle_fn(current_task)
            diff = bundle.diff or ""
            if not diff.strip():
                reason = f"ciclo {cycle} sin diff: nada que commitear"
                break
            summary = diff_summary(diff)
            if self.verbose:
                print("=" * 65)
                print(f"[MACRO] Ciclo {cycle}/{self.max_cycles} | {summary}")
            if not self._consent_commit(cycle, summary, branch, bundle.files):
                print("[MACRO] Commit declinado por el usuario. Diff mostrado, sin commitear:")
                print(diff)
                reason = f"commit declinado en ciclo {cycle}"
                break
            try:
                commit_diff_to_branch(
                    self.project_path, branch, original, current_task, diff, cycle=cycle,
                )
                commits += 1
            except GitBridgeError as e:
                err_text = e.stderr or str(e)
                if bundle.repair_chat is None:
                    print(f"[ERROR GIT] {e}")
                    reason = f"git falló en ciclo {cycle} sin reparación"
                    break
                if self.verbose:
                    print("[MACRO] Apply falló, intentando auto-reparación (1 intento)...")
                try:
                    fixed = bundle.repair_chat(
                        build_repair_prompt(current_task, err_text, bundle.repair_context)
                    )
                except ProviderError as pe:
                    print(f"[ERROR] Auto-reparación falló: {pe}")
                    reason = f"reparación falló en ciclo {cycle}"
                    break
                try:
                    commit_diff_to_branch(
                        self.project_path, branch, original, current_task, fixed, cycle=cycle,
                    )
                    commits += 1
                    diff = fixed
                except GitBridgeError as e2:
                    print(f"[ERROR GIT] La reparación también falló: {e2}")
                    reason = f"git falló tras reparación en ciclo {cycle}"
                    break
            history_lines.append(f"c{cycle}: {summary} | tarea: {current_task[:80]}")
            self.detector.observe(bundle.files, current_task)
            if cycle >= self.max_cycles:
                reason = f"techo max-cycles={self.max_cycles} alcanzado"
                finished = False
                break
            history = "\n".join(history_lines[-3:])
            estado, finding = self._audit(bundle, current_task, cycle, history)
            if estado != AUDIT_FINDING or not finding:
                finished = True
                reason = "TAREA_FINALIZADA por micro-auditoría"
                break
            loop, why = self.detector.is_oscillation(bundle.files, finding)
            # El detector también debe frenar si el hallazgo repite archivos+tema previos
            if loop:
                print(f"[AVISO] Detección de oscilación/bucle entre ciclos. Forzando cierre seguro. ({why})")
                findings.append(finding)
                finished = False
                reason = f"ping-pong detectado: {why}"
                break
            findings.append(finding)
            if not self._consent_continue(cycle, finding):
                reason = f"usuario frenó cascada tras ciclo {cycle}"
                finished = False
                break
            current_task = (
                f"Tarea original: {self.initial_task}. "
                f"Hallazgo tras ciclo {cycle}: {finding}. "
                f"Continúa desde el estado actual del disco."
            )
        else:
            reason = reason or "fin de cascada"
        if commits == 0 and branch:
            from git_bridge import delete_branch
            delete_branch(self.project_path, branch)
            branch = ""
        pushed = False
        if commits and self.push and branch:
            try:
                push_branch(self.project_path, branch)
                pushed = True
            except GitBridgeError as e:
                print(f"[ERROR GIT push] {e}")
                return 1, MacroResult(
                    branch=branch, original_branch=original, cycles_used=cycles_used,
                    commits=commits, findings=findings, finished=False,
                    stopped_reason=f"{reason} + push falló", pushed=False,
                )
        if self.verbose:
            print("=" * 65)
            print(f"[MACRO] Ciclos: {cycles_used} | Commits: {commits} | Rama: {branch or '(sin rama: 0 commits)'}")
            if branch:
                print(f"Revisa con: git diff {original}...{branch}")
            if findings:
                print(f"Hallazgos en cadena: {' -> '.join(findings)}")
            print(f"Cierre: {reason}")
        return 0, MacroResult(
            branch=branch, original_branch=original, cycles_used=cycles_used,
            commits=commits, findings=findings,
            finished=finished,
            stopped_reason=reason, pushed=pushed,
        )
