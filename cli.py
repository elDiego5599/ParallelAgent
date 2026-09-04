#!/usr/bin/env python3
"""Punto de entrada de ParallelAgent (CLI).

Parsea argumentos, infiere la topología (peer vs lead), valida incompatibilidades
y despacha la ejecución al motor correspondiente.
"""

import argparse
from pathlib import Path
import sys
from typing import Literal

from orchestrator import PeerEngine
from lead_engine import LeadEngine
from providers import parse_model_spec, participant_labels, strip_alias


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parallel-agent",
        description="ParallelAgent: Sistema de deliberación multi-modelo para código.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Requeridos
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Descripción técnica de la tarea a resolver.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Ruta al repositorio de código objetivo.",
    )

    # Topología PEER
    peer_group = parser.add_argument_group("Topología PEER (Mesa redonda)")
    peer_group.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Lista de modelos en igualdad de condiciones (mínimo 2).\n"
        "Ej: --models claude-3-7-sonnet gpt-4o deepseek-r1\n"
        "Gemelos: --models opus opus (auto: 'opus (1)', 'opus (2)')\n"
        "Alias: --models opus=arquitecto opus=auditor",
    )
    peer_group.add_argument(
        "--writer",
        type=str,
        default=None,
        metavar="MODEL",
        help="Modelo encargado de transcribir el diff final (solo modo peer).\n"
        "Acepta spec, alias o slug: 'opus=arquitecto', 'arquitecto' u 'opus'. "
        "Por defecto: último en hablar.",
    )

    # Topología LEAD
    lead_group = parser.add_argument_group(
        "Topología LEAD (Líder + Asesores)"
    )
    lead_group.add_argument(
        "--lead",
        type=str,
        metavar="MODEL",
        help="Modelo líder que propone, refina y emite el código.\n"
        "Acepta alias: --lead opus=líder",
    )
    lead_group.add_argument(
        "--advisors",
        nargs="+",
        metavar="MODEL",
        help="Modelos revisores que auditan, objetan o vetan (mínimo 1).\n"
        "Ej: --advisors gpt-4o deepseek-r1\n"
        "Gemelos: --advisors opus opus (auto-desambiguados)",
    )

    # Opciones de ejecución comunes
    common_group = parser.add_argument_group("Opciones generales")
    common_group.add_argument(
        "--mode",
        choices=["build", "plan", "ask"],
        default="build",
        help="Modo de ejecución (por defecto: build):\n"
        "  build: delibera y aplica diff a rama de Git.\n"
        "  plan:  delibera y devuelve especificación técnica.\n"
        "  ask:   consulta técnica o auditoría sin tocar código.",
    )
    common_group.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        metavar="N",
        help="Límite máximo de rondas de deliberación (por defecto: 4).",
    )
    common_group.add_argument(
        "--quorum",
        choices=["unanime", "mayoria"],
        default="unanime",
        help="Criterio de parada para topología peer (por defecto: unanime).",
    )
    common_group.add_argument(
        "--non-interactive",
        action="store_true",
        help="No preguntar por terminal. Ante PREGUNTA_AL_USUARIO se asume "
        "la vía conservadora (útil en CI/CD).",
    )
    common_group.add_argument(
        "--context-budget",
        type=int,
        default=12000,
        metavar="N",
        help="Presupuesto de caracteres del mapa de contexto (por defecto: 12000). "
        "Bájalo si tu tier limita tokens por minuto.",
    )
    common_group.add_argument(
        "--max-cycles",
        type=int,
        default=1,
        metavar="N",
        help="Ciclos máximos de cascada condicional solo en modo build (por defecto: 1).\n"
        "  1: clásico, un debate + un commit sin micro-auditoría.\n"
        "  >1: tras cada commit, micro-auditoría barata; solo se expande con NUEVO_HALLAZGO.\n"
        "  Tope de seguridad, no meta.",
    )
    common_group.add_argument(
        "--yes",
        action="store_true",
        help="Auto-aprueba commits de la cascada sin preguntar (útil en CI).\n"
        "Sin --yes y en interactivo se pide consentimiento por ciclo.",
    )
    common_group.add_argument(
        "--push",
        action="store_true",
        help="Push explícito de la rama consensus a origin al terminar (por defecto: no).\n"
        "El texto de la tarea nunca dispara push por sí solo.",
    )


    return parser


def validate_and_infer_topology(
    args: argparse.Namespace,
) -> Literal["peer", "lead"]:
    """Valida la consistencia de flags y determina la topología activa.

    Lanza ValueError con mensajes claros ante inconsistencias.
    """
    # 1. Validación de ruta
    if not args.path.exists():
        raise ValueError(f"La ruta especificada no existe: {args.path}")
    if not args.path.is_dir():
        raise ValueError(
            f"La ruta debe ser un directorio de proyecto válido: {args.path}"
        )

    # 2. Validación de rondas y presupuesto
    if args.max_rounds < 1:
        raise ValueError("--max-rounds debe ser un entero mayor o igual a 1.")
    if args.context_budget < 1000:
        raise ValueError("--context-budget debe ser un entero mayor o igual a 1000.")
    if args.max_cycles < 1:
        raise ValueError("--max-cycles debe ser un entero mayor o igual a 1.")
    if args.push and args.mode != "build":
        raise ValueError("--push solo tiene sentido en modo build (plan/ask no tocan Git).")
    if args.max_cycles > 1 and args.mode != "build":
        raise ValueError("--max-cycles > 1 solo aplica en modo build (sin disco no hay cascada).")

    has_peer = bool(args.models)
    has_lead = bool(args.lead or args.advisors)

    # 3. Detección de colisión entre topologías
    if has_peer and has_lead:
        raise ValueError(
            "Conflicto de topologías: no puedes mezclar '--models' (peer) con '--lead/--advisors' (lead).\n"
            "Usa uno u otro según el esquema de deliberación deseado."
        )

    if not has_peer and not has_lead:
        raise ValueError(
            "Debes definir un comité usando topología peer (--models M1 M2 ...) "
            "o topología lead (--lead M1 --advisors M2 ...)."
        )

    # 4. Validaciones específicas de PEER
    if has_peer:
        if len(args.models) < 2:
            raise ValueError(
                "La topología peer requiere al menos 2 modelos para deliberar (--models M1 M2)."
            )

        if args.writer:
            labels = participant_labels(args.models)
            apis = [strip_alias(m) for m in args.models]
            specs = list(args.models)
            w_api, w_alias = parse_model_spec(args.writer)
            ok = (
                args.writer in specs
                or args.writer in labels
                or args.writer in apis
                or (w_alias and w_alias in labels)
                or (w_api in apis)
            )
            if not ok:
                raise ValueError(
                    f"El redactor indicado (--writer '{args.writer}') no forma parte de la mesa: {args.models}"
                )

        return "peer"

    # 5. Validaciones específicas de LEAD
    if has_lead:
        if not args.lead or not args.advisors:
            raise ValueError(
                "La topología lead requiere definir tanto '--lead' (un modelo) como '--advisors' (mínimo un modelo)."
            )

        if args.writer:
            raise ValueError(
                "El flag '--writer' es inválido en topología lead.\n"
                f"El redactor es siempre y exclusivamente el líder ('{args.lead}')."
            )

        lead_api = strip_alias(args.lead)
        advisor_apis = [strip_alias(a) for a in args.advisors]
        if lead_api in advisor_apis:
            raise ValueError(
                f"Conflicto de roles: el modelo '{lead_api}' no puede ser líder y asesor a la vez."
            )

        return "lead"

    raise RuntimeError("Estado de topología no alcanzable.")


def print_banner(topology: str, args: argparse.Namespace) -> None:
    """Muestra un resumen visual estructurado antes de iniciar la sesión."""
    print("=" * 65)
    print(
        f" ParallelAgent | Topología: {topology.upper()} | Modo: {args.mode.upper()}"
    )
    print("=" * 65)
    print(f"Tarea:       {args.task}")
    print(f"Proyecto:    {args.path.resolve()}")
    print(f"Rondas max:  {args.max_rounds}")
    print(f"Ciclos max:  {args.max_cycles}")
    print(f"Push:        {'sí (--push)' if args.push else 'no'}")
    print(f"Interactivo: {'no (--non-interactive)' if args.non_interactive else 'sí'}")
    if not args.yes and not args.non_interactive and args.max_cycles > 1:
        print("Consenso Git:  se pedirá confirmación por ciclo (usa --yes para auto-aprobar)")

    if topology == "peer":
        labels = participant_labels(args.models)
        writer_label = (
            f"{args.writer} (fijado)"
            if args.writer
            else "Último en hablar (dinámico)"
        )
        print(f"Quorum:      {args.quorum}")
        print(f"Redactor:    {writer_label}")
        print("Participantes:")
        for spec, lab in zip(args.models, labels):
            api, alias = parse_model_spec(spec)
            extra = f" [slug API: {api}]" if alias else ""
            print(f"    - {lab}{extra}")
    else:
        lead_labels = participant_labels([args.lead, *args.advisors])
        print(f"Lider (Tech Lead): {lead_labels[0]}")
        print("Asesores (Reviewers):")
        for lab, spec in zip(lead_labels[1:], args.advisors):
            api, alias = parse_model_spec(spec)
            extra = f" [slug API: {api}]" if alias else ""
            print(f"    - {lab}{extra}")
    print("=" * 65 + "\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        topology = validate_and_infer_topology(args)
    except ValueError as err:
        print(f"\n[ERROR DE CONFIGURACIÓN] {err}\n", file=sys.stderr)
        return 2

    print_banner(topology, args)

    # Despacho al motor correspondiente
    if topology == "peer":
        engine = PeerEngine(
            task=args.task,
            project_path=args.path,
            models=args.models,
            writer=args.writer,
            mode=args.mode,
            max_rounds=args.max_rounds,
            quorum=args.quorum,
            interactive=not args.non_interactive,
            context_budget=args.context_budget,
        )
        if args.mode == "build" and args.max_cycles > 1:
            return _run_macro_peer(engine, args)
        return engine.run()

    elif topology == "lead":
        engine = LeadEngine(
            task=args.task,
            project_path=args.path,
            lead=args.lead,
            advisors=args.advisors,
            mode=args.mode,
            max_rounds=args.max_rounds,
            interactive=not args.non_interactive,
            context_budget=args.context_budget,
        )
        if args.mode == "build" and args.max_cycles > 1:
            return _run_macro_lead(engine, args)
        return engine.run()

    return 0


def _run_macro_peer(engine, args) -> int:
    from macro_engine import CycleBundle, MacroEngine

    def cycle_fn(task: str) -> CycleBundle:
        result, providers, labels, context = engine.run_cycle(task, verbose=True)
        by_label = dict(zip(labels, providers))
        wp = by_label.get(result.writer, providers[-1])
        wl = result.writer

        def audit_chat(prompt, _wp=wp, _wl=wl):
            return _wp.chat([
                {"role": "system", "content": (
                    f"Eres {_wl}. Auditor post-ciclo: solo estado, sin código.")},
                {"role": "user", "content": prompt},
            ])

        def repair_chat(prompt, _wp=wp, _wl=wl):
            return _wp.chat([
                {"role": "system", "content": (
                    f"Eres {_wl}. Corriges el diff como redactor, "
                    "sin añadir decisiones nuevas.")},
                {"role": "user", "content": prompt},
            ])

        return CycleBundle(
            result=result,
            diff=result.final_output or "",
            files=list(result.files_declared or []),
            writer_label=wl,
            repair_context=result.base_context or context,
            audit_chat=audit_chat,
            repair_chat=repair_chat,
        )

    macro = MacroEngine(
        project_path=args.path,
        initial_task=args.task,
        max_cycles=args.max_cycles,
        cycle_fn=cycle_fn,
        interactive=not args.non_interactive,
        auto_approve=args.yes,
        push=args.push,
    )
    rc, _ = macro.run()
    return rc


def _run_macro_lead(engine, args) -> int:
    from macro_engine import CycleBundle, MacroEngine

    def cycle_fn(task: str) -> CycleBundle:
        result, lead, _advisors, labels, context = engine.run_cycle(task, verbose=True)
        lead_label = labels["lead"] if isinstance(labels, dict) else result.writer

        def audit_chat(prompt, _lead=lead, _lab=lead_label):
            return _lead.chat([
                {"role": "system", "content": (
                    f"Eres {_lab}. Auditor post-ciclo: solo estado, sin código.")},
                {"role": "user", "content": prompt},
            ])

        def repair_chat(prompt, _lead=lead, _lab=lead_label):
            return _lead.chat([
                {"role": "system", "content": (
                    f"Eres {_lab}. Corriges el diff como líder, "
                    "sin añadir decisiones nuevas.")},
                {"role": "user", "content": prompt},
            ])

        return CycleBundle(
            result=result,
            diff=result.final_output or "",
            files=list(result.files_declared or []),
            writer_label=lead_label,
            repair_context=result.base_context or context,
            audit_chat=audit_chat,
            repair_chat=repair_chat,
        )

    macro = MacroEngine(
        project_path=args.path,
        initial_task=args.task,
        max_cycles=args.max_cycles,
        cycle_fn=cycle_fn,
        interactive=not args.non_interactive,
        auto_approve=args.yes,
        push=args.push,
    )
    rc, _ = macro.run()
    return rc


if __name__ == "__main__":
    sys.exit(main())
