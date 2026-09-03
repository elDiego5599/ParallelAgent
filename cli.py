#!/usr/bin/env python3
"""Punto de entrada de ParallelAgent (CLI).

Parsea argumentos, infiere la topología (peer vs lead), valida incompatibilidades
y despacha la ejecución al motor correspondiente.
"""

import argparse
from pathlib import Path
import sys
from typing import Literal

from orchestrator import LeadEngine, PeerEngine


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
        "Ej: --models claude-3-7-sonnet gpt-4o deepseek-r1",
    )
    peer_group.add_argument(
        "--writer",
        type=str,
        default=None,
        metavar="MODEL",
        help="Modelo encargado de transcribir el diff final (solo modo peer).\n"
        "Debe pertenecer a --models. Por defecto: último en hablar.",
    )

    # Topología LEAD
    lead_group = parser.add_argument_group(
        "Topología LEAD (Líder + Asesores)"
    )
    lead_group.add_argument(
        "--lead",
        type=str,
        metavar="MODEL",
        help="Modelo líder que propone, refina y emite el código.",
    )
    lead_group.add_argument(
        "--advisors",
        nargs="+",
        metavar="MODEL",
        help="Modelos revisores que auditan, objetan o vetan (mínimo 1).\n"
        "Ej: --advisors gpt-4o deepseek-r1",
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

    # 2. Validación de rondas
    if args.max_rounds < 1:
        raise ValueError("--max-rounds debe ser un entero mayor o igual a 1.")

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

        if args.writer and args.writer not in args.models:
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

        if args.lead in args.advisors:
            raise ValueError(
                f"Conflicto de roles: el modelo '{args.lead}' no puede ser líder y asesor a la vez."
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

    if topology == "peer":
        writer_label = (
            f"{args.writer} (fijado)"
            if args.writer
            else "Último en hablar (dinámico)"
        )
        print(f"Quorum:      {args.quorum}")
        print(f"Redactor:    {writer_label}")
        print("Participantes:")
        for m in args.models:
            print(f"    - {m}")
    else:
        print(f"Lider (Tech Lead): {args.lead}")
        print("Asesores (Reviewers):")
        for a in args.advisors:
            print(f"    - {a}")
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
        )
        return engine.run()

    elif topology == "lead":
        engine = LeadEngine(
            task=args.task,
            project_path=args.path,
            lead=args.lead,
            advisors=args.advisors,
            mode=args.mode,
            max_rounds=args.max_rounds,
        )
        return engine.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
