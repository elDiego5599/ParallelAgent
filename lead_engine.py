"""Motor de topología lead para ParallelAgent.

Un modelo lidera (propone, refina y redacta) y el resto audita sin escribir
código. Reutiliza providers, contexto, HITL y salida a Git del motor peer.
Solo cambia la máquina de estados y la alternancia de turnos.
"""

from pathlib import Path
import re
from typing import Callable, Dict, List, Optional

from context import build_repo_context
from orchestrator import (
    DEBATING,
    QUESTION,
    SYSTEM,
    DebateResult,
    Turn,
    _resolve_question,
    build_emission_prompt,
    finish_build_output,
)
from providers import BaseProvider, ProviderError, resolve_provider, warn_unknown_models


CONFORME = "CONFORME"
OBJECTION = "OBJECION_BLOQUEANTE"
VETO = "VETO_ARQUITECTONICO"

_LEAD_RE = re.compile(
    r"[*_]{0,2}\s*ESTADO:\s*(CONFORME|DEBATIENDO|OBJECION_BLOQUEANTE|VETO_ARQUITECTONICO|PREGUNTA_AL_USUARIO)[*_]{0,2}",
    re.IGNORECASE,
)


def parse_lead_estado(text: str) -> str:
    """Extrae el marcador del asesor. Si falta o viene malformado, es DEBATIENDO."""
    matches = _LEAD_RE.findall(text or "")
    if not matches:
        return DEBATING
    return matches[-1].upper()


def build_lead_system_prompt(lead: str, advisors: List[str], task: str, mode: str) -> str:
    names = ", ".join(advisors)
    return (
        f"Eres {lead}, el líder técnico de esta mesa de revisión. Tus asesores son: {names}.\n"
        f"Tarea: {task}\n"
        f"Modo de salida acordado: {mode}.\n"
        "REGLAS DEL LÍDER:\n"
        "1. Propón la solución base y defiéndela o corrígela ante las críticas.\n"
        "2. Eres el único autorizado a redactar código o diffs. Nadie más lo hará.\n"
        "3. Responde punto por punto las objeciones bloqueantes antes de avanzar.\n"
        "4. Si tu premisa es vetada por unanimidad, replantea el diseño desde cero.\n"
        "5. Cierra con ESTADO: DEBATIENDO. Usa ESTADO: PREGUNTA_AL_USUARIO solo ante "
        "un bloqueo de negocio imposible de deducir del código (formato PREGUNTA: ...). "
        "Prohibido preguntar por estilo o convenciones."
    )


def build_advisor_system_prompt(advisor: str, lead: str, task: str) -> str:
    return (
        f"Eres {advisor}, revisor senior. El líder técnico es {lead}.\n"
        f"Tarea bajo revisión: {task}\n"
        "REGLAS DEL ASESOR:\n"
        "1. Tienes prohibido escribir código ejecutable. Solo auditas.\n"
        "2. Busca fugas, condiciones de carrera, APIs rotas y supuestos no probados.\n"
        "3. Cierra cada mensaje con un estado exacto:\n"
        "   ESTADO: CONFORME (apruebas sin objeciones),\n"
        "   ESTADO: DEBATIENDO (sugerencias menores),\n"
        "   ESTADO: OBJECION_BLOQUEANTE (fallo crítico que el líder debe responder),\n"
        "   ESTADO: VETO_ARQUITECTONICO (la premisa base es inválida y debe descartarse),\n"
        "   ESTADO: PREGUNTA_AL_USUARIO (bloqueo de negocio, formato PREGUNTA: ...).\n"
        "4. Usa VETO solo si la base es una alucinación o mala práctica grave, no por desacuerdos menores."
    )


def build_lead_turn_prompt(
    task: str,
    context: str,
    transcript: List[Turn],
    round_num: int,
    objections: List[str],
    vetoed: bool,
) -> str:
    lines = [f"[Ronda {round_num}] [LÍDER]", f"Tarea: {task}"]
    if context and round_num == 1:
        lines.append(f"Contexto del repositorio:\n{context}")
    if vetoed:
        lines.append(
            "Tu premisa fue VETADA POR UNANIMIDAD. Replantea el diseño desde cero, "
            "sin insistir en la propuesta descartada."
        )
    if objections:
        lines.append("Objeciones bloqueantes pendientes de tu respuesta:")
        for o in objections:
            lines.append(f"  - {o[:500]}")
    if transcript:
        lines.append("Transcripción hasta ahora:")
        for t in transcript[-8:]:
            lines.append(f"--- [{t.model}] (ronda {t.round}, {t.estado}) ---\n{t.text}")
    elif round_num == 1:
        lines.append("Presenta tu propuesta de solución base.")
    lines.append("Responde o propón de forma breve. Cierra con ESTADO: DEBATIENDO.")
    return "\n\n".join(lines)


def build_advisor_turn_prompt(task: str, lead_text: str, round_num: int) -> str:
    return (
        f"[Ronda {round_num}] [REVISIÓN]\n"
        f"Tarea: {task}\n"
        f"Última propuesta del líder:\n{lead_text}\n\n"
        "Audita con rigor y cierra con tu estado (CONFORME, DEBATIENDO, "
        "OBJECION_BLOQUEANTE, VETO_ARQUITECTONICO o PREGUNTA_AL_USUARIO)."
    )


def run_lead_debate(
    lead: BaseProvider,
    advisors: List[BaseProvider],
    task: str,
    context: str = "",
    mode: str = "build",
    max_rounds: int = 4,
    verbose: bool = True,
    interactive: bool = True,
    ask_user: Optional[Callable[[str, str], str]] = None,
    max_questions: int = 3,
) -> DebateResult:
    if not advisors:
        raise ValueError("Se requiere al menos 1 asesor.")
    if mode not in ("build", "plan", "ask"):
        raise ValueError(f"Modo desconocido: {mode}")

    advisor_ids = [a.model_id for a in advisors]
    result = DebateResult(task=task, mode=mode, writer=lead.model_id)
    questions_asked = 0
    pending_objections: List[str] = []
    vetoed = False

    lead_system = build_lead_system_prompt(lead.model_id, advisor_ids, task, mode)
    advisor_systems: Dict[str, str] = {
        a.model_id: build_advisor_system_prompt(a.model_id, lead.model_id, task)
        for a in advisors
    }

    def say(model: str, text: str, estado: str, round_num: int) -> None:
        result.transcript.append(Turn(round=round_num, model=model, text=text, estado=estado))
        if verbose:
            print(f"\n[{model} | ronda {round_num} | {estado}]\n{text}\n")

    def maybe_ask(model: str, text: str, round_num: int) -> None:
        nonlocal questions_asked
        _resolve_question(
            result, round_num, model, text,
            interactive, ask_user, questions_asked, max_questions, verbose,
        )
        questions_asked += 1

    for round_num in range(1, max_rounds + 1):
        try:
            lead_text = lead.chat([
                {"role": "system", "content": lead_system},
                {"role": "user", "content": build_lead_turn_prompt(
                    task, context, result.transcript, round_num,
                    pending_objections, vetoed)},
            ])
        except ProviderError as e:
            lead_text = f"Error de proveedor: {e}\n\nESTADO: DEBATIENDO"
        lead_estado = parse_lead_estado(lead_text)
        say(lead.model_id, lead_text, DEBATING, round_num)
        if lead_estado == QUESTION:
            maybe_ask(lead.model_id, lead_text, round_num)

        advisor_estados: List[str] = []
        pending_objections = []
        for advisor in advisors:
            try:
                text = advisor.chat([
                    {"role": "system", "content": advisor_systems[advisor.model_id]},
                    {"role": "user", "content": build_advisor_turn_prompt(task, lead_text, round_num)},
                ])
            except ProviderError as e:
                text = f"Error de proveedor: {e}\n\nESTADO: DEBATIENDO"
            estado = parse_lead_estado(text)
            say(advisor.model_id, text, estado, round_num)
            if estado == QUESTION:
                maybe_ask(advisor.model_id, text, round_num)
                estado = DEBATING
            if estado == OBJECTION:
                pending_objections.append(f"{advisor.model_id}: {text[:500]}")
            advisor_estados.append(estado)

        result.rounds_used = round_num
        vetoed = bool(advisor_estados) and all(e == VETO for e in advisor_estados)
        if vetoed:
            note = (
                "VETO UNÁNIME: todos los asesores rechazan la premisa base. "
                "El líder debe replantear el diseño desde cero en la próxima ronda."
            )
            result.transcript.append(Turn(round=round_num, model=SYSTEM, text=note, estado=DEBATING))
            if verbose:
                print(f"\n[{SYSTEM} -> mesa]\n{note}\n")
            if round_num >= max_rounds:
                break
            continue
        if all(e == CONFORME for e in advisor_estados):
            result.consensus_reached = True
            break
        if round_num >= max_rounds:
            break

    emission = [
        {"role": "system", "content": (
            f"Eres {lead.model_id}. Redactas el resultado final como líder, "
            f"respetando lo acordado. Modo: {mode}.")},
        {"role": "user", "content": build_emission_prompt(task, result.transcript, mode, context)},
    ]
    try:
        result.final_output = lead.chat(emission)
    except ProviderError as e:
        result.final_output = f"Error en emisión ({lead.model_id}): {e}"

    return result


class LeadEngine:
    """Motor de topología lead. Líder propone y redacta, asesores auditan."""

    def __init__(
        self,
        task: str,
        project_path: Path,
        lead: str,
        advisors: List[str],
        mode: str = "build",
        max_rounds: int = 4,
        context: str = "",
        context_budget: int = 12000,
        interactive: bool = True,
    ):
        self.task = task
        self.project_path = project_path
        self.lead_id = lead
        self.advisor_ids = advisors
        self.mode = mode
        self.max_rounds = max_rounds
        self.context = context
        self.context_budget = context_budget
        self.interactive = interactive

    def run(self) -> int:
        warn_unknown_models([self.lead_id, *self.advisor_ids])
        try:
            lead = resolve_provider(self.lead_id)
            advisor_providers = [resolve_provider(a) for a in self.advisor_ids]
        except ProviderError as e:
            print(f"[ERROR] {e}")
            return 1
        if lead.model_id in [a.model_id for a in advisor_providers]:
            print(f"[ERROR] '{lead.model_id}' no puede ser líder y asesor a la vez.")
            return 2
        context = self.context or build_repo_context(
            self.project_path, self.task, max_total_chars=self.context_budget
        )
        print(f"Contexto: {len(context)} caracteres del repositorio.")
        result = run_lead_debate(
            lead,
            advisor_providers,
            task=self.task,
            context=context,
            mode=self.mode,
            max_rounds=self.max_rounds,
            verbose=True,
            interactive=self.interactive,
        )
        print("=" * 65)
        print(
            f"Consenso: {result.consensus_reached} | "
            f"Rondas: {result.rounds_used} | Líder: {result.writer}"
        )
        print("=" * 65)
        if self.mode == "build":
            def repair(prompt, _lead=lead):
                return _lead.chat([
                    {"role": "system", "content": (
                        f"Eres {lead.model_id}. Corriges el diff como líder, "
                        "sin añadir decisiones nuevas.")},
                    {"role": "user", "content": prompt},
                ])
            return finish_build_output(
                self.project_path, self.task, result.final_output, context, repair
            )
        print(result.final_output)
        return 0 if result.final_output else 1
