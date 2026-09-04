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
    MAX_ACUERDOS,
    QUESTION,
    SYSTEM,
    DebateResult,
    Turn,
    _resolve_question,
    _short,
    build_emission_prompt,
    finish_build_output,
    record_acuerdo,
    resolve_multifile_context,
)
from providers import (
    BaseProvider,
    ProviderError,
    parse_model_spec,
    participant_labels,
    resolve_provider,
    warn_unknown_models,
)


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
    acuerdos: Optional[List[str]] = None,
) -> str:
    lines = [f"[Ronda {round_num}] [LÍDER]", f"Tarea: {task}"]
    if context and round_num == 1:
        lines.append(f"Contexto del repositorio:\n{context}")
    if acuerdos:
        lines.append(
            "ACUERDOS_PREVIOS (objeciones ya resueltas; no reabrir sin motivo nuevo):\n"
            + "\n".join(f"- {a}" for a in acuerdos[-MAX_ACUERDOS:])
        )
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


def build_advisor_turn_prompt(
    task: str, lead_text: str, round_num: int, acuerdos: Optional[List[str]] = None,
) -> str:
    base = (
        f"[Ronda {round_num}] [REVISIÓN]\n"
        f"Tarea: {task}\n"
        f"Última propuesta del líder:\n{lead_text}\n\n"
        "Audita con rigor y cierra con tu estado (CONFORME, DEBATIENDO, "
        "OBJECION_BLOQUEANTE, VETO_ARQUITECTONICO o PREGUNTA_AL_USUARIO)."
    )
    if acuerdos:
        base += (
            "\n\nACUERDOS_PREVIOS (ya resueltos; no reabrir sin motivo nuevo):\n"
            + "\n".join(f"- {a}" for a in acuerdos[-MAX_ACUERDOS:])
        )
    return base


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
    project_path: Optional[Path] = None,
    enable_multifile: bool = True,
    lead_label: Optional[str] = None,
    advisor_labels: Optional[List[str]] = None,
) -> DebateResult:
    if not advisors:
        raise ValueError("Se requiere al menos 1 asesor.")
    if mode not in ("build", "plan", "ask"):
        raise ValueError(f"Modo desconocido: {mode}")

    all_specs = [lead.model_id] + [a.model_id for a in advisors]
    joint = participant_labels(all_specs)
    if not lead_label:
        lead_label = joint[0]
    if advisor_labels is None or len(advisor_labels) != len(advisors):
        advisor_labels = joint[1:]
    result = DebateResult(task=task, mode=mode, writer=lead_label)
    questions_asked = 0
    pending_objections: List[str] = []
    open_objections: Dict[str, str] = {}
    vetoed = False

    lead_system = build_lead_system_prompt(lead_label, advisor_labels, task, mode)
    advisor_systems: Dict[str, str] = {
        lab: build_advisor_system_prompt(lab, lead_label, task)
        for lab in advisor_labels
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
                    pending_objections, vetoed, result.acuerdos)},
            ])
        except ProviderError as e:
            lead_text = f"Error de proveedor: {e}\n\nESTADO: DEBATIENDO"
        lead_estado = parse_lead_estado(lead_text)
        say(lead_label, lead_text, DEBATING, round_num)
        if lead_estado == QUESTION:
            maybe_ask(lead_label, lead_text, round_num)

        advisor_estados: List[str] = []
        pending_objections = []
        for idx, advisor in enumerate(advisors):
            lab = advisor_labels[idx]
            try:
                text = advisor.chat([
                    {"role": "system", "content": advisor_systems[lab]},
                    {"role": "user", "content": build_advisor_turn_prompt(
                        task, lead_text, round_num, result.acuerdos)},
                ])
            except ProviderError as e:
                text = f"Error de proveedor: {e}\n\nESTADO: DEBATIENDO"
            estado = parse_lead_estado(text)
            say(lab, text, estado, round_num)
            if estado == QUESTION:
                maybe_ask(lab, text, round_num)
                estado = DEBATING
            if estado == OBJECTION:
                pending_objections.append(f"{lab}: {text[:500]}")
                open_objections[lab] = _short(text, 120)
            elif estado == CONFORME and lab in open_objections:
                record_acuerdo(
                    result,
                    f"{lab} objetó '{open_objections.pop(lab)}' → conforme en R{round_num} (resuelto, no reabrir).",
                )
            advisor_estados.append(estado)

        result.rounds_used = round_num
        vetoed = bool(advisor_estados) and all(e == VETO for e in advisor_estados)
        if vetoed:
            note = (
                "VETO UNÁNIME: todos los asesores rechazan la premisa base. "
                "El líder debe replantear el diseño desde cero en la próxima ronda."
            )
            result.transcript.append(Turn(round=round_num, model=SYSTEM, text=note, estado=DEBATING))
            record_acuerdo(result, f"R{round_num}: veto unánime; el líder replantea desde cero.")
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

    emission = None
    declared: List[str] = []
    base_files = ""
    if mode == "build" and enable_multifile:
        declared, base_files = resolve_multifile_context(
            lead, task, result.transcript, project_path, verbose=False,
            display_name=lead_label,
        )
        result.files_declared = declared
        result.base_context = base_files
        if verbose and declared:
            print(f"[MULTIFILE] {lead_label} declara {len(declared)} archivo(s).")
    emission = [
        {"role": "system", "content": (
            f"Eres {lead_label}. Redactas el resultado final como líder, "
            f"respetando lo acordado. Modo: {mode}.")},
        {"role": "user", "content": build_emission_prompt(task, result.transcript, mode, context, base_files)},
    ]
    try:
        result.final_output = lead.chat(emission)
    except ProviderError as e:
        result.final_output = f"Error en emisión ({lead_label}): {e}"

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

    def run_cycle(self, task: Optional[str] = None, verbose: bool = True):
        """Un ciclo de deliberación SIN efectos Git. Devuelve (result, lead, advisors, labels, context)."""
        warn_unknown_models([self.lead_id, *self.advisor_ids])
        lead_api, _ = parse_model_spec(self.lead_id)
        advisor_apis = [parse_model_spec(a)[0] for a in self.advisor_ids]
        lead = resolve_provider(lead_api)
        advisor_providers = [resolve_provider(a) for a in advisor_apis]
        if lead_api in advisor_apis:
            raise ValueError(f"'{lead_api}' no puede ser líder y asesor a la vez.")
        joint = participant_labels([self.lead_id, *self.advisor_ids])
        lead_label, advisor_labels = joint[0], joint[1:]
        cycle_task = task if task is not None else self.task
        context = self.context or build_repo_context(
            self.project_path, cycle_task, max_total_chars=self.context_budget
        )
        result = run_lead_debate(
            lead,
            advisor_providers,
            task=cycle_task,
            context=context,
            mode=self.mode,
            max_rounds=self.max_rounds,
            verbose=verbose,
            interactive=self.interactive,
            project_path=self.project_path,
            lead_label=lead_label,
            advisor_labels=advisor_labels,
        )
        labels = {"lead": lead_label, "advisors": advisor_labels}
        return result, lead, advisor_providers, labels, context

    def run(self) -> int:
        try:
            result, lead, advisor_providers, labels, context = self.run_cycle()
        except (ProviderError, ValueError) as e:
            print(f"[ERROR] {e}")
            return 2 if "asesor" in str(e) else 1
        lead_label, advisor_labels = labels["lead"], labels["advisors"]
        print(f"Contexto: {len(context)} caracteres del repositorio.")
        print(f"Mesa: líder {lead_label} + {', '.join(advisor_labels)}")
        print("=" * 65)
        print(
            f"Consenso: {result.consensus_reached} | "
            f"Rondas: {result.rounds_used} | Líder: {result.writer}"
        )
        if result.files_declared:
            print(f"Archivos: {', '.join(result.files_declared)}")
        print("=" * 65)
        if self.mode == "build":
            def repair(prompt, _lead=lead, _lab=lead_label):
                return _lead.chat([
                    {"role": "system", "content": (
                        f"Eres {_lab}. Corriges el diff como líder, "
                        "sin añadir decisiones nuevas.")},
                    {"role": "user", "content": prompt},
                ])
            repair_context = result.base_context or context
            return finish_build_output(
                self.project_path, self.task, result.final_output, repair_context, repair
            )
        print(result.final_output)
        return 0 if result.final_output else 1
