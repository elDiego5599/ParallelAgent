"""Gemelos y alias: anti-espejo (labels únicas, API intacta)."""

import pytest

from cli import build_parser, validate_and_infer_topology
from lead_engine import run_lead_debate
from orchestrator import run_debate
from providers import (
    BaseProvider,
    ProviderError,
    parse_model_spec,
    participant_labels,
    resolve_provider,
    strip_alias,
)


def parse(tmp_path, *argv):
    return build_parser().parse_args(["--task", "t", "--path", str(tmp_path), *argv])


# --- parsing ---

def test_parse_model_spec_igual_y_dos_puntos():
    assert parse_model_spec("opus=arquitecto") == ("opus", "arquitecto")
    assert parse_model_spec("openrouter/x=auditor") == ("openrouter/x", "auditor")
    assert parse_model_spec("opus:auditor") == ("opus", "auditor")
    assert parse_model_spec("groq/llama=rapido") == ("groq/llama", "rapido")


def test_parse_preserva_mock():
    assert parse_model_spec("mock") == ("mock", None)
    assert parse_model_spec("mock:2") == ("mock:2", None)
    assert parse_model_spec("mock:3:llama") == ("mock:3:llama", None)
    assert strip_alias("mock:2") == "mock:2"


def test_strip_alias():
    assert strip_alias("opus=arquitecto") == "opus"
    assert strip_alias("opus:auditor") == "opus"
    assert strip_alias("groq/llama=rapido") == "groq/llama"
    assert strip_alias("opus") == "opus"


def test_participant_labels_unicos_intactos():
    assert participant_labels(["a", "b"]) == ["a", "b"]
    assert participant_labels(["mock:2:a", "mock:2:b"]) == ["mock:2:a", "mock:2:b"]


def test_participant_labels_gemelos_auto():
    assert participant_labels(["opus", "opus"]) == ["opus (1)", "opus (2)"]
    assert participant_labels(["x", "x", "x"]) == ["x (1)", "x (2)", "x (3)"]


def test_participant_labels_alias():
    assert participant_labels(["opus=arquitecto", "opus=auditor"]) == ["arquitecto", "auditor"]
    # Mismo alias dos veces -> sufijo para no colapsar
    assert participant_labels(["opus=x", "claude=x"]) == ["x (1)", "x (2)"]


def test_resolve_pela_alias_antes_de_api():
    # Sin keys: debe fallar por falta de key del slug pelado, no del spec con alias
    with pytest.raises(ProviderError) as exc:
        resolve_provider("groq/llama=rapido")
    assert "llama" in str(exc.value) and "=rapido" not in str(exc.value)


# --- peer con gemelos funcionales (no solo psicológicos) ---

class Twin(BaseProvider):
    def __init__(self, mid="opus"):
        super().__init__(mid)
        self.seen_peers = []

    def chat(self, messages):
        sys = messages[0].get("content", "")
        # Captura con quién cree que debate
        for line in sys.splitlines():
            if "colegas:" in line.lower():
                self.seen_peers.append(line)
        return "De acuerdo.\nESTADO: CONSENSO_ALCANZADO"


def test_peer_gemelos_no_colapsan_y_peers_no_vacios():
    t1, t2 = Twin("opus"), Twin("opus")
    r = run_debate([t1, t2], task="t", max_rounds=1, verbose=False)
    models = [t.model for t in r.transcript if t.round == 1]
    assert models == ["opus (1)", "opus (2)"]
    # Sin fix: by_id colapsaba a 1 y peers=[] ; ahora cada uno ve al otro
    assert t1.seen_peers and "opus (2)" in t1.seen_peers[0]
    assert t2.seen_peers and "opus (1)" in t2.seen_peers[0]
    assert r.consensus_reached and r.rounds_used == 1
    assert r.writer in ("opus (1)", "opus (2)")


def test_peer_writer_acepta_slug_api_ambiguo():
    r = run_debate(
        [Twin("opus"), Twin("opus")], task="t", max_rounds=1,
        writer="opus", verbose=False,
    )
    assert r.writer == "opus (1)"  # primero ante ambigüedad


def test_peer_labels_explicitas_alias():
    a = Twin("api-opus")
    b = Twin("api-opus")
    r = run_debate(
        [a, b], task="t", max_rounds=1, verbose=False,
        labels=["arquitecto", "auditor"], writer="auditor",
    )
    assert [t.model for t in r.transcript if t.round == 1] == ["arquitecto", "auditor"]
    assert r.writer == "auditor"


# --- lead con asesores gemelos ---

class ConformeTwin(BaseProvider):
    def __init__(self, mid="opus"):
        super().__init__(mid)

    def chat(self, messages):
        return "LGTM.\nESTADO: CONFORME"


class LeadTwin(BaseProvider):
    def __init__(self, mid="opus-lead"):
        super().__init__(mid)

    def chat(self, messages):
        last = messages[-1].get("content", "").lower()
        if "diff unificado" in last:
            return "DIFF"
        return "Propuesta.\nESTADO: DEBATIENDO"


def test_lead_asesores_gemelos_fast_path():
    r = run_lead_debate(
        LeadTwin(), [ConformeTwin("opus"), ConformeTwin("opus")],
        task="t", mode="plan", verbose=False,
    )
    advisors = [t.model for t in r.transcript if t.model in ("opus (1)", "opus (2)")]
    assert advisors == ["opus (1)", "opus (2)"]
    assert r.consensus_reached


# --- CLI alias-aware ---

def test_cli_peer_gemelos_y_writer_slug(tmp_path):
    args = parse(tmp_path, "--models", "opus", "opus", "--writer", "opus")
    assert validate_and_infer_topology(args) == "peer"


def test_cli_peer_alias_writer(tmp_path):
    args = parse(tmp_path, "--models", "opus=arquitecto", "opus=auditor", "--writer", "arquitecto")
    assert validate_and_infer_topology(args) == "peer"


def test_cli_peer_writer_fuera_rechaza(tmp_path):
    with pytest.raises(ValueError, match="no forma parte"):
        validate_and_infer_topology(
            parse(tmp_path, "--models", "opus=arquitecto", "opus=auditor", "--writer", "otro")
        )


def test_cli_lead_mismo_slug_con_alias_se_rechaza(tmp_path):
    with pytest.raises(ValueError, match="no puede ser líder y asesor"):
        validate_and_infer_topology(
            parse(tmp_path, "--lead", "opus=lider", "--advisors", "opus=auditor")
        )


def test_cli_lead_gemelos_asesores_ok(tmp_path):
    args = parse(tmp_path, "--lead", "claude", "--advisors", "opus", "opus")
    assert validate_and_infer_topology(args) == "lead"
