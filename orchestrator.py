"""Orquestador del debate para ParallelAgent.

No opina sobre el código. Reparte turnos, mantiene la transcripción
compartida y aplica la regla de parada por quórum.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional

from context import build_repo_context
from git_bridge import GitBridgeError, apply_consensus_to_branch
from providers import (
    BaseProvider,
    ProviderError,
    parse_model_spec,
    participant_labels,
    resolve_provider,
    warn_unknown_models,
)


HUMAN = "HUMANO / TECH LEAD"
SYSTEM = "SISTEMA"


CONSENSUS = "CONSENSO_ALCANZADO"
DEBATING = "DEBATIENDO"
QUESTION = "PREGUNTA_AL_USUARIO"

_ESTADO_RE = re.compile(
    r"[*_]{0,2}\s*ESTADO:\s*(DEBATIENDO|CONSENSO_ALCANZADO|PREGUNTA_AL_USUARIO)[*_]{0,2}",
    re.IGNORECASE,
)


def parse_estado(text: str) -> str:
    """Extrae el marcador final. Si falta o viene malformado, es DEBATIENDO."""
    matches = _ESTADO_RE.findall(text or "")
    if not matches:
        return DEBATING
    return matches[-1].upper()


@dataclass
class Turn:
    round: int
    model: str
    text: str
    estado: str


@dataclass
class DebateResult:
    task: str
    mode: str
    transcript: List[Turn] = field(default_factory=list)
    consensus_reached: bool = False
    rounds_used: int = 0
    final_output: str = ""
    writer: str = ""
    files_declared: List[str] = field(default_factory=list)
    base_context: str = ""


def build_system_prompt(model_id: str, peers: List[str], task: str, mode: str) -> str:
    colleagues = ", ".join(peers) if peers else "ninguno"
    return (
        f"Eres {model_id}, un arquitecto de software de élite.\n"
        f"Estás en una mesa de trabajo técnica junto a tus colegas: {colleagues}.\n"
        f"Tu objetivo grupal es resolver la siguiente tarea técnica con cero errores: {task}\n"
        f"Modo de salida acordado: {mode}.\n"
        "REGLAS DE LA MESA:\n"
        "1. Habla directamente a tus colegas cuando discrepes o confirmes.\n"
        "2. Cuestiona con rigor técnico: condiciones de carrera, fugas, APIs deprecadas.\n"
        "3. No seas complaciente: si una propuesta falla, señálalo con código mínimo.\n"
        "4. En deliberación no reescribas archivos completos, solo diseño y riesgos.\n"
        "5. Cierra cada mensaje con una línea exacta: ESTADO: DEBATIENDO, ESTADO: CONSENSO_ALCANZADO o ESTADO: PREGUNTA_AL_USUARIO.\n"
        "Usa CONSENSO_ALCANZADO solo si estás de acuerdo con la propuesta vigente de la mesa.\n"
        "6. Usa PREGUNTA_AL_USUARIO solo ante un bloqueo arquitectónico o de negocio imposible de deducir del código. "
        "Formula la duda como PREGUNTA: ... Prohibido preguntar por estilo, nombres o convenciones."
    )


def _window_transcript(transcript: List[Turn], round_num: int) -> tuple:
    """Ventana deslizante: conserva la ronda 1 (pitches) y desde la anterior.

    Descarta la charla intermedia para mantener el payload acotado sin importar
    la cantidad de rondas. Devuelve (turnos, omitidos).
    """
    keep = [t for t in transcript if t.round == 1 or t.round >= round_num - 1]
    return keep, len(transcript) - len(keep)


def build_turn_prompt(
    task: str,
    context: str,
    transcript: List[Turn],
    round_num: int,
    mode: str,
) -> str:
    lines = [f"[Ronda {round_num}]", f"Tarea: {task}", f"Modo: {mode}"]
    if context:
        lines.append(f"Contexto del repositorio:\n{context}")
    if transcript:
        lines.append("Transcripción hasta ahora (ventana deslizante):")
        windowed, omitted = _window_transcript(transcript, round_num)
        for t in windowed:
            lines.append(f"--- [{t.model}] (ronda {t.round}, {t.estado}) ---\n{t.text}")
        if omitted:
            lines.append(
                f"[... {omitted} intervenciones intermedias omitidas por ventana ...]"
            )
    else:
        lines.append("Eres el primero en hablar. Presenta tu pitch inicial.")
    lines.append(
        "Intervén de forma breve. Responde a lo dicho por la mesa y "
        "cierra con ESTADO: DEBATIENDO, ESTADO: CONSENSO_ALCANZADO o, "
        "solo ante un bloqueo real, ESTADO: PREGUNTA_AL_USUARIO."
    )
    return "\n\n".join(lines)


def build_emission_prompt(
    task: str,
    transcript: List[Turn],
    mode: str,
    context: str = "",
    base_files: str = "",
) -> str:
    body = "\n\n".join(f"[{t.model}]:\n{t.text}" for t in transcript)
    if mode == "plan":
        instruction = (
            "Transcribe el acuerdo de la mesa como plan técnico en Markdown: "
            "objetivo, diseño propuesto, archivos afectados, riesgos y pasos. "
            "Sin código completo, sin diff."
        )
    elif mode == "ask":
        instruction = (
            "Transcribe la respuesta técnica acordada por la mesa de forma directa y completa. "
            "Sin diff, sin crear ramas."
        )
    else:
        instruction = (
            "Transcribe el acuerdo de la mesa como parche en formato diff unificado "
            "contra el contenido original de abajo, con líneas de contexto y "
            "cabeceras de hunk correctas. Por cada archivo incluye:\n"
            "diff --git a/<ruta> b/<ruta>\n"
            "--- a/<ruta>\n"
            "+++ b/<ruta>\n"
            "Solo el diff crudo, sin cercas Markdown."
        )
    prompt = f"Tarea: {task}\n\nAcuerdo de la mesa:\n{body}\n\n{instruction}"
    if mode == "build" and context:
        prompt += f"\n\nContenido original de referencia:\n{context}"
    if mode == "build" and base_files:
        prompt += (
            "\n\nContenido base exacto de los archivos a modificar "
            "(fuente de verdad para rutas y hunks):\n"
            f"{base_files}\n\n"
            "Genera un único bloque diff unificado que contenga los cambios "
            "para cada uno de ellos, con cabeceras --- a/<ruta> +++ b/<ruta> "
            "correctas y rutas exactas declaradas."
        )
    return prompt


# --- Emisión multi-archivo (Vector 1): Declarar -> Reinyectar -> Diff ---

MAX_DECLARE_FILES = 10
MAX_BASE_FILE_CHARS = 8000
MAX_BASE_TOTAL_CHARS = 30000

_FILE_DECLARE_MARKER = "lista únicamente las rutas relativas exactas"


def build_file_declaration_prompt(task: str) -> str:
    return (
        f"Tarea: {task}\n\n"
        "La mesa ha alcanzado consenso. Antes de generar el diff, "
        "lista únicamente las rutas relativas exactas de los archivos del "
        "repositorio que necesitas modificar o crear.\n"
        "Una ruta por línea, sin explicaciones ni markdown. Ejemplo:\n"
        "lib/bridge.dart\n"
        "android/app/src/main/kotlin/MainActivity.kt"
    )


def _parse_file_list(response: str) -> List[str]:
    """Limpia la declaración del redactor a rutas relativas seguras.

    Tolera viñetas (`-`, `*`, `1.`), backticks y citas. Rechaza absolutas,
    traversals (`..`), líneas explicativas y basura sin forma de ruta.
    """
    paths: List[str] = []
    for raw in (response or "").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        # Quita citas, viñetas y numeración: "> - 1. ruta"
        s = re.sub(r"^(?:>\s*|[-*+•]\s*|\d+[.\)]\s*)+", "", s.strip())
        s = s.strip().strip("`'\"").strip()
        s = s.rstrip(".,:;")
        if not s or len(s) > 200:
            continue
        if " " in s and "/" not in s:
            continue  # frase explicativa, no ruta
        if not re.fullmatch(r"[A-Za-z0-9_.\-/]+", s):
            continue
        if "/" not in s and "." not in s:
            continue
        p = Path(s)
        if p.is_absolute() or ".." in p.parts:
            continue
        if s not in paths:
            paths.append(s)
        if len(paths) >= MAX_DECLARE_FILES:
            break
    return paths


def resolve_participants(specs: List[str]) -> tuple:
    """Resuelve specs con alias a (providers, labels) sin colapsar gemelos.

    'opus=arquitecto' -> provider opus + label 'arquitecto'.
    ['opus', 'opus'] -> labels ['opus (1)', 'opus (2)'].
    """
    apis = [parse_model_spec(s)[0] for s in specs]
    providers = [resolve_provider(a) for a in apis]
    labels = participant_labels(specs)
    return providers, labels


def ask_files_to_modify(
    writer_provider: BaseProvider,
    task: str,
    transcript: Optional[List[Turn]] = None,
    display_name: Optional[str] = None,
) -> List[str]:
    """Micro-paso 1 (~50 tokens): el redactor declara qué archivos tocará.

    Ante cualquier fallo del proveedor devuelve [] para que el llamador
    use el fallback mono-contexto. Nunca lanza.
    """
    _ = transcript  # reservado: el redactor ya vivió el debate, no reinyectamos todo
    who = display_name or writer_provider.model_id
    messages = [
        {
            "role": "system",
            "content": (
                f"Eres {who}. Declaras archivos, no código. "
                "Respondes solo con rutas relativas, una por línea."
            ),
        },
        {"role": "user", "content": build_file_declaration_prompt(task)},
    ]
    try:
        response = writer_provider.chat(messages)
    except ProviderError:
        return []
    except Exception:
        return []
    return _parse_file_list(response)


def build_multi_file_base_context(
    project_path: Path,
    relative_paths: List[str],
    max_file_chars: int = MAX_BASE_FILE_CHARS,
    max_total_chars: int = MAX_BASE_TOTAL_CHARS,
) -> str:
    """Micro-paso 2 determinista: lee del disco el contenido exacto declarado.

    Los archivos inexistentes se marcan como nuevos (partir desde vacío).
    Respeta topes por archivo y total para no volar el TPM.
    """
    blocks: List[str] = []
    total = 0
    for rel in relative_paths[:MAX_DECLARE_FILES]:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            blocks.append(f"--- RUTA OMITIDA (fuera del repo): {rel} ---")
            continue
        fp = project_path / rel
        try:
            is_file = fp.is_file()
        except OSError:
            is_file = False
        if is_file:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                blocks.append(f"--- ARCHIVO ILEGIBLE: {rel} ---")
                continue
            if len(content) > max_file_chars:
                content = (
                    content[:max_file_chars]
                    + f"\n\n[... Truncado: original {len(content)} chars ...]"
                )
            block = f"--- ARCHIVO EXISTENTE: {rel} ---\n```\n{content}\n```"
        else:
            block = f"--- ARCHIVO NUEVO A CREAR: {rel} --- (partir desde vacío)"
        if total + len(block) > max_total_chars:
            room = max_total_chars - total
            if room > 500:
                blocks.append(block[:room] + "\n[... Truncado por límite total ...]")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def resolve_multifile_context(
    writer_provider: BaseProvider,
    task: str,
    transcript: List[Turn],
    project_path: Optional[Path],
    verbose: bool = False,
    display_name: Optional[str] = None,
) -> tuple:
    """Ejecuta Declarar -> Reinyectar. Devuelve (declared, base_context).

    Fallback seguro: ([], "") si no hay project_path o el redactor no
    declara nada útil. Nunca lanza.
    """
    if project_path is None:
        return [], ""
    try:
        declared = ask_files_to_modify(writer_provider, task, transcript, display_name)
    except Exception:
        return [], ""
    if not declared:
        return [], ""
    try:
        base = build_multi_file_base_context(project_path, declared)
    except Exception:
        return [], ""
    if verbose and declared:
        print(f"[MULTIFILE] Archivos declarados ({len(declared)}): {', '.join(declared)}")
        print(f"[MULTIFILE] Base exacta: {len(base)} caracteres.")
    return declared, base


def _quorum_met(estados: List[str], quorum: str) -> bool:
    n = len(estados)
    agreed = sum(1 for e in estados if e == CONSENSUS)
    if quorum == "mayoria":
        return agreed >= max(1, n - 1)
    return n > 0 and agreed == n


def _extract_question(text: str) -> str:
    """Limpia el marcador de estado y devuelve la duda formulada."""
    clean = _ESTADO_RE.sub("", text or "").strip()
    return clean[-1200:] if len(clean) > 1200 else clean


def default_ask_user(model: str, question: str) -> str:
    print("=" * 65)
    print("[ParallelAgent] LA MESA SOLICITA ACLARACIÓN TÉCNICA")
    print("=" * 65)
    print(f"Modelo: {model}")
    print(f"Duda:   {question}")
    try:
        return input("\nTu respuesta > ").strip()
    except EOFError:
        return ""


def _resolve_question(
    result: DebateResult,
    round_num: int,
    model: str,
    text: str,
    interactive: bool,
    ask_user: Optional[Callable[[str, str], str]],
    questions_asked: int,
    max_questions: int,
    verbose: bool,
) -> None:
    """Pausa la sala, consigue la aclaración y la inyecta al transcript."""
    if questions_asked >= max_questions:
        note = "Límite de preguntas alcanzado. Continúen con la alternativa más conservadora."
        source = SYSTEM
    elif interactive:
        handler = ask_user or default_ask_user
        answer = handler(model, _extract_question(text))
        if not answer.strip():
            note = "Sin respuesta. Asuman la alternativa más conservadora y continúen."
            source = SYSTEM
        else:
            note = answer
            source = HUMAN
    else:
        note = (
            "El usuario no está disponible (modo no interactivo). "
            "Asuman la alternativa más conservadora y continúen."
        )
        source = SYSTEM
    result.transcript.append(
        Turn(round=round_num, model=source, text=note, estado=DEBATING)
    )
    if verbose:
        print(f"\n[{source} -> mesa]\n{note}\n")


def run_debate(
    providers: List[BaseProvider],
    task: str,
    context: str = "",
    mode: str = "build",
    max_rounds: int = 4,
    quorum: str = "unanime",
    writer: Optional[str] = None,
    verbose: bool = True,
    interactive: bool = True,
    ask_user: Optional[Callable[[str, str], str]] = None,
    max_questions: int = 3,
    project_path: Optional[Path] = None,
    enable_multifile: bool = True,
    labels: Optional[List[str]] = None,
) -> DebateResult:
    if len(providers) < 2:
        raise ValueError("Se requieren al menos 2 modelos en la mesa.")
    if mode not in ("build", "plan", "ask"):
        raise ValueError(f"Modo desconocido: {mode}")
    if quorum not in ("unanime", "mayoria"):
        raise ValueError(f"Quórum desconocido: {quorum}")

    if labels is None or len(labels) != len(providers):
        labels = participant_labels([p.model_id for p in providers])
    by_label: Dict[str, BaseProvider] = dict(zip(labels, providers))
    label_of = dict(zip([id(p) for p in providers], labels))

    result = DebateResult(task=task, mode=mode)
    last_speaker = labels[-1]
    questions_asked = 0

    for round_num in range(1, max_rounds + 1):
        round_estados: List[str] = []
        for idx, provider in enumerate(providers):
            me = labels[idx]
            peers = [l for j, l in enumerate(labels) if j != idx]
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(me, peers, task, mode),
                },
                {
                    "role": "user",
                    "content": build_turn_prompt(task, context, result.transcript, round_num, mode),
                },
            ]
            try:
                text = provider.chat(messages)
            except ProviderError as e:
                text = f"Error de proveedor: {e}\n\nESTADO: DEBATIENDO"
            estado = parse_estado(text)
            result.transcript.append(
                Turn(round=round_num, model=me, text=text, estado=estado)
            )
            if estado == QUESTION:
                if verbose:
                    print(f"\n[{me} | ronda {round_num} | {estado}]\n{text}\n")
                _resolve_question(
                    result, round_num, me, text,
                    interactive, ask_user, questions_asked, max_questions, verbose,
                )
                questions_asked += 1
                estado = DEBATING
            elif verbose:
                print(f"\n[{me} | ronda {round_num} | {estado}]\n{text}\n")
            round_estados.append(estado)
            last_speaker = me

        result.rounds_used = round_num
        if _quorum_met(round_estados, quorum):
            result.consensus_reached = True
            break

    writer_id = None
    writer_provider = None
    if writer in by_label:
        writer_id = writer
        writer_provider = by_label[writer]
    else:
        apis = [p.model_id for p in providers]
        if writer in apis:
            idx = apis.index(writer)
            writer_id = labels[idx]
            writer_provider = providers[idx]
        else:
            writer_id = last_speaker
            writer_provider = by_label[writer_id]
    _ = label_of
    declared: List[str] = []
    base_files = ""
    if mode == "build" and enable_multifile:
        declared, base_files = resolve_multifile_context(
            writer_provider, task, result.transcript, project_path,
            verbose=False, display_name=writer_id,
        )
        result.files_declared = declared
        result.base_context = base_files
        if verbose and declared:
            print(f"[MULTIFILE] {writer_id} declara {len(declared)} archivo(s).")
    emission_messages = [
        {
            "role": "system",
            "content": (
                f"Eres {writer_id}. Transcribes fielmente el acuerdo de la mesa, "
                f"sin añadir decisiones nuevas. Modo: {mode}."
            ),
        },
        {
            "role": "user",
            "content": build_emission_prompt(task, result.transcript, mode, context, base_files),
        },
    ]
    try:
        result.final_output = writer_provider.chat(emission_messages)
    except ProviderError as e:
        result.final_output = f"Error en emisión ({writer_id}): {e}"
    result.writer = writer_id

    return result


def _report_branch(info: dict) -> None:
    print(f"Rama: {info['branch']}")
    print(f"Revisa con: {info['inspect']}")
    if info["dirty_before"]:
        print("[AVISO] El árbol tenía cambios sin commitear antes de empezar.")


def build_repair_prompt(task: str, git_stderr: str, context: str = "") -> str:
    """Prompt de auto-reparación: error exacto de git + contenido original."""
    parts = [
        f"Tarea: {task}",
        "El parche diff que generaste falló al aplicarse con 'git apply'.",
        "Detalle exacto del error de Git:",
        "-" * 50,
        git_stderr,
        "-" * 50,
    ]
    if context:
        parts.append(f"Contenido original de referencia:\n{context}")
    parts.append(
        "INSTRUCCIONES DE CORRECCIÓN: Analiza el error (líneas no coincidentes, "
        "hunks desalineados u offsets). Emite EXCLUSIVAMENTE el diff unificado "
        "corregido, crudo y completo. Sin cercas Markdown, sin explicaciones."
    )
    return "\n\n".join(parts)


def finish_build_output(
    project_path: Path,
    task: str,
    final_output: str,
    context: str = "",
    repair_chat=None,
) -> int:
    """Aplica el diff; ante fallo, da UNA oportunidad de auto-reparación al redactor.

    Los abortos por árbol sucio son fatales: no se reintentan porque el
    problema no es el diff sino el WIP del usuario.
    """
    if not final_output:
        return 1
    try:
        _report_branch(apply_consensus_to_branch(project_path, task, final_output))
        return 0
    except GitBridgeError as e:
        err_text = e.stderr or str(e)
        if "sin commitear" in str(e).lower() or "sin commitear" in err_text.lower():
            print(f"[ERROR GIT] {e}")
            return 1
        if repair_chat is None:
            print(f"[ERROR GIT] {e}\nDiff emitido (no aplicado):")
            print(final_output)
            return 1
    print("[BUILD] Parche inicial falló con error de Git. Solicitando auto-reparación...")
    try:
        fixed = repair_chat(build_repair_prompt(task, err_text, context))
    except ProviderError as pe:
        print(f"[ERROR] Auto-reparación falló: {pe}")
        return 1
    try:
        _report_branch(apply_consensus_to_branch(project_path, task, fixed))
        print("[BUILD] Auto-reparación aplicada.")
        return 0
    except GitBridgeError as e2:
        print(f"[ERROR GIT] La reparación también falló: {e2}\nDiff reparado (no aplicado):")
        print(fixed)
        return 1


class PeerEngine:
    """Motor de topología peer. Resuelve proveedores y ejecuta run_debate."""

    def __init__(
        self,
        task: str,
        project_path: Path,
        models: List[str],
        writer: Optional[str] = None,
        mode: str = "build",
        max_rounds: int = 4,
        quorum: str = "unanime",
        context: str = "",
        context_budget: int = 12000,
        interactive: bool = True,
    ):
        self.task = task
        self.project_path = project_path
        self.models = models
        self.writer = writer
        self.mode = mode
        self.max_rounds = max_rounds
        self.quorum = quorum
        self.context = context
        self.context_budget = context_budget
        self.interactive = interactive

    def run(self) -> int:
        warn_unknown_models(self.models)
        try:
            providers = [resolve_provider(m) for m in self.models]
        except ProviderError as e:
            print(f"[ERROR] {e}")
            return 1
        context = self.context or build_repo_context(
            self.project_path, self.task, max_total_chars=self.context_budget
        )
        print(f"Contexto: {len(context)} caracteres del repositorio.")
        result = run_debate(
            providers,
            task=self.task,
            context=context,
            mode=self.mode,
            max_rounds=self.max_rounds,
            quorum=self.quorum,
            writer=self.writer,
            verbose=True,
            interactive=self.interactive,
        )
        print("=" * 65)
        print(
            f"Consenso: {result.consensus_reached} | "
            f"Rondas: {result.rounds_used} | Redactor: {result.writer}"
        )
        print("=" * 65)
        if self.mode == "build":
            by_id = {p.model_id: p for p in providers}
            writer_provider = by_id.get(result.writer)
            repair = None
            if writer_provider is not None:
                def repair(prompt, _wp=writer_provider, _w=result.writer):
                    return _wp.chat([
                        {"role": "system", "content": (
                            f"Eres {_w}. Corriges el diff como redactor, "
                            "sin añadir decisiones nuevas.")},
                        {"role": "user", "content": prompt},
                    ])
            repair_context = result.base_context or context
            return finish_build_output(
                self.project_path, self.task, result.final_output, repair_context, repair
            )
        print(result.final_output)
        return 0 if result.final_output else 1
