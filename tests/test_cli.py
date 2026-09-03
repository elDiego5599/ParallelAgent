"""CLI: inferencia de topología y matriz de incompatibilidades."""

import pytest

from cli import build_parser, validate_and_infer_topology


def parse(tmp_path, *argv):
    return build_parser().parse_args(["--task", "t", "--path", str(tmp_path), *argv])


def test_peer_ok(tmp_path):
    args = parse(tmp_path, "--models", "a", "b", "--writer", "a")
    assert validate_and_infer_topology(args) == "peer"


def test_peer_needs_two_models(tmp_path):
    with pytest.raises(ValueError, match="al menos 2 modelos"):
        validate_and_infer_topology(parse(tmp_path, "--models", "solo-uno"))


def test_peer_writer_must_belong(tmp_path):
    with pytest.raises(ValueError, match="no forma parte de la mesa"):
        validate_and_infer_topology(parse(tmp_path, "--models", "a", "b", "--writer", "c"))


def test_topologies_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="Conflicto de topologías"):
        validate_and_infer_topology(
            parse(tmp_path, "--models", "a", "b", "--lead", "a", "--advisors", "b")
        )


def test_committee_is_required(tmp_path):
    with pytest.raises(ValueError, match="Debes definir un comité"):
        validate_and_infer_topology(parse(tmp_path))


def test_lead_needs_both_sides(tmp_path):
    with pytest.raises(ValueError, match="tanto '--lead'"):
        validate_and_infer_topology(parse(tmp_path, "--lead", "a"))
    args = parse(tmp_path, "--lead", "a", "--advisors", "b")
    assert validate_and_infer_topology(args) == "lead"


def test_writer_forbidden_in_lead(tmp_path):
    with pytest.raises(ValueError, match="'--writer' es inválido"):
        validate_and_infer_topology(
            parse(tmp_path, "--lead", "a", "--advisors", "b", "--writer", "a")
        )


def test_lead_cannot_advise_itself(tmp_path):
    with pytest.raises(ValueError, match="no puede ser líder y asesor"):
        validate_and_infer_topology(
            parse(tmp_path, "--lead", "a", "--advisors", "a", "b")
        )


def test_path_must_exist_and_be_dir(tmp_path):
    with pytest.raises(ValueError, match="no existe"):
        validate_and_infer_topology(
            parse(tmp_path, "--models", "a", "b", "--path", str(tmp_path / "nope"))
        )
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="directorio"):
        validate_and_infer_topology(parse(tmp_path, "--models", "a", "b", "--path", str(f)))


def test_max_rounds_floor(tmp_path):
    with pytest.raises(ValueError, match="--max-rounds"):
        validate_and_infer_topology(
            parse(tmp_path, "--models", "a", "b", "--max-rounds", "0")
        )


def test_non_interactive_flag(tmp_path):
    assert parse(tmp_path, "--models", "a", "b").non_interactive is False
    assert parse(tmp_path, "--models", "a", "b", "--non-interactive").non_interactive is True
