"""Motor peer: quórum, redactor, HITL y regresión."""

from providers import BaseProvider, resolve_provider
from orchestrator import QUESTION, run_debate, parse_estado


def test_parse_estado():
    assert parse_estado("x\nESTADO: CONSENSO_ALCANZADO") == "CONSENSO_ALCANZADO"
    assert parse_estado("x\nestado: debatiendo") == "DEBATIENDO"
    assert parse_estado("x\nESTADO: PREGUNTA_AL_USUARIO") == QUESTION
    assert parse_estado("sin marcador") == "DEBATIENDO"


def test_unanimous_closes_when_slowest_agrees():
    ps = [
        resolve_provider("mock:2:a"),
        resolve_provider("mock:2:b"),
        resolve_provider("mock:3:c"),
    ]
    r = run_debate(ps, task="t", max_rounds=4, writer="mock:2:a", verbose=False)
    assert r.rounds_used == 3
    assert r.consensus_reached is True
    assert r.writer == "mock:2:a"
    assert len(r.transcript) == 9


def test_mayoria_needs_n_minus_1():
    ps = [resolve_provider("mock:2:a"), resolve_provider("mock:2:b")]
    r = run_debate(ps, task="t", max_rounds=4, quorum="mayoria", verbose=False)
    assert r.consensus_reached and r.rounds_used == 2


def test_writer_fallback_to_last_speaker():
    ps = [resolve_provider("mock"), resolve_provider("mock:2:b")]
    r = run_debate(ps, task="t", max_rounds=1, writer="inexistente", verbose=False)
    assert r.writer == "mock:2:b"
    assert r.consensus_reached is False


def test_modes_emit():
    for mode in ("plan", "ask", "build"):
        ps = [resolve_provider("mock"), resolve_provider("mock:2:b")]
        r = run_debate(ps, task="t", mode=mode, max_rounds=2, verbose=False)
        assert r.final_output and r.mode == mode


class Questioner(BaseProvider):
    def __init__(self, mid, ask_first=True):
        super().__init__(mid)
        self.n = 0
        self.ask_first = ask_first

    def chat(self, messages):
        self.n += 1
        if self.ask_first and self.n == 1:
            return "Dos diseños. PREGUNTA: singleton o dispose?\nESTADO: PREGUNTA_AL_USUARIO"
        return "De acuerdo.\nESTADO: CONSENSO_ALCANZADO"


def test_hitl_injects_human_answer_and_continues():
    asked = []
    r = run_debate(
        [Questioner("qa"), Questioner("qb", ask_first=False)],
        task="t",
        max_rounds=3,
        interactive=True,
        ask_user=lambda m, q: (asked.append((m, q)), "Singleton.")[1],
        verbose=False,
    )
    assert len(asked) == 1 and asked[0][0] == "qa"
    human = [t for t in r.transcript if t.model == "HUMANO / TECH LEAD"]
    assert len(human) == 1 and human[0].text == "Singleton."
    assert r.consensus_reached and r.rounds_used == 2


def test_hitl_non_interactive_assumes_conservative():
    r = run_debate(
        [Questioner("qa"), Questioner("qb", ask_first=False)],
        task="t",
        max_rounds=3,
        interactive=False,
        ask_user=lambda m, q: (_ for _ in ()).throw(AssertionError("no debe preguntar")),
        verbose=False,
    )
    assert any(t.model == "SISTEMA" and "conservadora" in t.text for t in r.transcript)
    assert r.consensus_reached


def test_hitl_question_cap():
    asked = []
    r = run_debate(
        [Questioner("qa"), Questioner("qb")],
        task="t",
        max_rounds=2,
        interactive=True,
        ask_user=lambda m, q: (asked.append(m), "x")[1],
        max_questions=1,
        verbose=False,
    )
    assert len(asked) == 1
    assert any("Límite de preguntas" in t.text for t in r.transcript)
