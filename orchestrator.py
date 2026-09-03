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
from providers import BaseProvider, ProviderError, resolve_provider


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
        lines.append("Transcripción hasta ahora:")
        for t in transcript:
            lines.append(f"--- [{t.model}] (ronda {t.round}, {t.estado}) ---\n{t.text}")
    else:
        lines.append("Eres el primero en hablar. Presenta tu pitch inicial.")
    lines.append(
        "Intervén de forma breve. Responde a lo dicho por la mesa y "
        "cierra con ESTADO: DEBATIENDO, ESTADO: CONSENSO_ALCANZADO o, "
        "solo ante un bloqueo real, ESTADO: PREGUNTA_AL_USUARIO."
    )
    return "\n\n".join(lines)


def build_emission_prompt(task: str, transcript: List[Turn], mode: str) -> str:
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
            "Transcribe el acuerdo de la mesa como parche en formato diff unificado, "
            "listo para aplicar. Sin explicaciones adicionales fuera del diff."
        )
    return (
        f"Tarea: {task}\n\nAcuerdo de la mesa:\n{body}\n\n{instruction}"
    )


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
) -> DebateResult:
    if len(providers) < 2:
        raise ValueError("Se requieren al menos 2 modelos en la mesa.")
    if mode not in ("build", "plan", "ask"):
        raise ValueError(f"Modo desconocido: {mode}")
    if quorum not in ("unanime", "mayoria"):
        raise ValueError(f"Quórum desconocido: {quorum}")

    ids = [p.model_id for p in providers]
    by_id: Dict[str, BaseProvider] = {p.model_id: p for p in providers}

    result = DebateResult(task=task, mode=mode)
    last_speaker = ids[-1]
    questions_asked = 0

    for round_num in range(1, max_rounds + 1):
        round_estados: List[str] = []
        for provider in providers:
            peers = [i for i in ids if i != provider.model_id]
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(provider.model_id, peers, task, mode),
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
                Turn(round=round_num, model=provider.model_id, text=text, estado=estado)
            )
            if estado == QUESTION:
                if verbose:
                    print(f"\n[{provider.model_id} | ronda {round_num} | {estado}]\n{text}\n")
                _resolve_question(
                    result, round_num, provider.model_id, text,
                    interactive, ask_user, questions_asked, max_questions, verbose,
                )
                questions_asked += 1
                estado = DEBATING
            elif verbose:
                print(f"\n[{provider.model_id} | ronda {round_num} | {estado}]\n{text}\n")
            round_estados.append(estado)
            last_speaker = provider.model_id

        result.rounds_used = round_num
        if _quorum_met(round_estados, quorum):
            result.consensus_reached = True
            break

    writer_id = writer if writer in by_id else last_speaker
    writer_provider = by_id[writer_id]
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
            "content": build_emission_prompt(task, result.transcript, mode),
        },
    ]
    try:
        result.final_output = writer_provider.chat(emission_messages)
    except ProviderError as e:
        result.final_output = f"Error en emisión ({writer_id}): {e}"
    result.writer = writer_id

    return result


def finish_build_output(project_path: Path, task: str, final_output: str) -> int:
    """Aplica el diff emitido sobre rama efímera. Común a PeerEngine y LeadEngine."""
    if not final_output:
        return 1
    try:
        info = apply_consensus_to_branch(project_path, task, final_output)
    except GitBridgeError as e:
        print(f"[ERROR GIT] {e}\nDiff emitido (no aplicado):")
        print(final_output)
        return 1
    print(f"Rama: {info['branch']}")
    print(f"Revisa con: {info['inspect']}")
    if info["dirty_before"]:
        print("[AVISO] El árbol tenía cambios sin commitear antes de empezar.")
    return 0


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
            return finish_build_output(self.project_path, self.task, result.final_output)
        print(result.final_output)
        return 0 if result.final_output else 1
