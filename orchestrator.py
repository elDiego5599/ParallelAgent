"""Orquestador del debate para ParalelAgent.

No opina sobre el código. Reparte turnos, mantiene la transcripción
compartida y aplica la regla de parada por quórum.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional

from providers import BaseProvider, ProviderError, resolve_provider


CONSENSUS = "CONSENSO_ALCANZADO"
DEBATING = "DEBATIENDO"

_ESTADO_RE = re.compile(r"ESTADO:\s*(DEBATIENDO|CONSENSO_ALCANZADO)", re.IGNORECASE)


def parse_estado(text: str) -> str:
    """Extrae el marcador final. Si falta o viene malformado, es DEBATIENDO."""
    matches = _ESTADO_RE.findall(text or "")
    if not matches:
        return DEBATING
    return CONSENSUS if matches[-1].upper() == CONSENSUS else DEBATING


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
        "5. Cierra cada mensaje con una línea exacta: ESTADO: DEBATIENDO o ESTADO: CONSENSO_ALCANZADO.\n"
        "Usa CONSENSO_ALCANZADO solo si estás de acuerdo con la propuesta vigente de la mesa."
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
        "cierra con ESTADO: DEBATIENDO o ESTADO: CONSENSO_ALCANZADO."
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


def run_debate(
    providers: List[BaseProvider],
    task: str,
    context: str = "",
    mode: str = "build",
    max_rounds: int = 4,
    quorum: str = "unanime",
    writer: Optional[str] = None,
    verbose: bool = True,
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
            round_estados.append(estado)
            last_speaker = provider.model_id
            if verbose:
                print(f"\n[{provider.model_id} | ronda {round_num} | {estado}]\n{text}\n")

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
    ):
        self.task = task
        self.project_path = project_path
        self.models = models
        self.writer = writer
        self.mode = mode
        self.max_rounds = max_rounds
        self.quorum = quorum
        self.context = context

    def run(self) -> int:
        try:
            providers = [resolve_provider(m) for m in self.models]
        except ProviderError as e:
            print(f"[ERROR] {e}")
            return 1
        result = run_debate(
            providers,
            task=self.task,
            context=self.context,
            mode=self.mode,
            max_rounds=self.max_rounds,
            quorum=self.quorum,
            writer=self.writer,
            verbose=True,
        )
        print("=" * 65)
        print(
            f"Consenso: {result.consensus_reached} | "
            f"Rondas: {result.rounds_used} | Redactor: {result.writer}"
        )
        print("=" * 65)
        print(result.final_output)
        return 0 if result.final_output else 1


class LeadEngine:
    """Motor de topología lead. Pendiente de implementación."""

    def __init__(
        self,
        task: str,
        project_path: Path,
        lead: str,
        advisors: List[str],
        mode: str = "build",
        max_rounds: int = 4,
    ):
        self.task = task
        self.project_path = project_path
        self.lead = lead
        self.advisors = advisors
        self.mode = mode
        self.max_rounds = max_rounds

    def run(self) -> int:
        print(
            "[AVISO] LeadEngine aún está en desarrollo. "
            "Ejecuta con --models (peer)."
        )
        return 1
