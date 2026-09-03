"""Motor lead: fast-path, objeciones, veto, HITL."""

import pytest

from providers import BaseProvider
from lead_engine import parse_lead_estado, run_lead_debate


def test_parse_lead_estados():
    assert parse_lead_estado("ok\nESTADO: CONFORME") == "CONFORME"
    assert parse_lead_estado("x\nESTADO: OBJECION_BLOQUEANTE") == "OBJECION_BLOQUEANTE"
    assert parse_lead_estado("x\nESTADO: VETO_ARQUITECTONICO") == "VETO_ARQUITECTONICO"
    assert parse_lead_estado("x\nESTADO: PREGUNTA_AL_USUARIO") == "PREGUNTA_AL_USUARIO"
    assert parse_lead_estado("ruido") == "DEBATIENDO"


class Advisor(BaseProvider):
    def __init__(self, mid, script):
        super().__init__(mid)
        self.script = list(script)
        self.n = 0

    def chat(self, messages):
        self.n += 1
        return self.script[min(self.n - 1, len(self.script) - 1)]


class Lead(BaseProvider):
    def __init__(self, mid="L"):
        super().__init__(mid)
        self.n = 0
        self.prompts = []

    def chat(self, messages):
        self.n += 1
        self.prompts.append(messages[-1]["content"])
        if any("diff unificado" in m.get("content", "") for m in messages):
            return "EMISION FINAL"
        return "Propuesta v%d.\nESTADO: DEBATIENDO" % self.n


AGREE = "ESTADO: CONFORME"
OBJECT = "Fallo en línea 12.\nESTADO: OBJECION_BLOQUEANTE"
VETO = "Premisa inválida.\nESTADO: VETO_ARQUITECTONICO"


def test_fast_path_closes_round_one():
    r = run_lead_debate(
        Lead(), [Advisor("a1", [AGREE]), Advisor("a2", [AGREE])],
        task="t", verbose=False,
    )
    assert r.consensus_reached and r.rounds_used == 1
    assert r.writer == "L" and r.final_output == "EMISION FINAL"


def test_objection_goes_back_to_leader():
    lead = Lead()
    r = run_lead_debate(
        lead, [Advisor("a1", [OBJECT, AGREE]), Advisor("a2", [AGREE, AGREE])],
        task="t", verbose=False,
    )
    assert r.consensus_reached and r.rounds_used == 2
    assert lead.n == 3  # 2 turnos de debate + emisión
    assert any("Objeciones bloqueantes" in p and "línea 12" in p for p in lead.prompts)


def test_unanimous_veto_forces_redesign():
    lead = Lead()
    r = run_lead_debate(
        lead, [Advisor("a1", [VETO, AGREE]), Advisor("a2", [VETO, AGREE])],
        task="t", verbose=False,
    )
    assert r.consensus_reached and r.rounds_used == 2
    assert any("VETO UNÁNIME" in t.text for t in r.transcript)
    assert any("VETADA POR UNANIMIDAD" in p for p in lead.prompts)


def test_partial_veto_is_not_unanimous():
    r = run_lead_debate(
        Lead(), [Advisor("a1", [VETO, AGREE]), Advisor("a2", [AGREE, AGREE])],
        task="t", verbose=False,
    )
    assert r.consensus_reached
    assert not any("VETO UNÁNIME" in t.text for t in r.transcript)


def test_lead_hitl():
    class Asker(BaseProvider):
        def __init__(self):
            super().__init__("aq")
            self.n = 0

        def chat(self, messages):
            self.n += 1
            if self.n == 1:
                return "PREGUNTA: singleton o dispose?\nESTADO: PREGUNTA_AL_USUARIO"
            return AGREE

    asked = []
    r = run_lead_debate(
        Lead(),
        [Asker(), Advisor("a2", [AGREE])],
        task="t",
        interactive=True,
        ask_user=lambda m, q: (asked.append(m), "Singleton")[1],
        verbose=False,
    )
    assert asked == ["aq"]
    assert any(t.model == "HUMANO / TECH LEAD" for t in r.transcript)
    assert r.consensus_reached


def test_requires_advisors():
    with pytest.raises(ValueError):
        run_lead_debate(Lead(), [], task="t")
