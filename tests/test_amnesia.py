"""Vector 2: ledger ACUERDOS_PREVIOS contra la amnesia de la ventana deslizante."""

from lead_engine import run_lead_debate
from orchestrator import (
    DebateResult,
    Turn,
    build_turn_prompt,
    record_acuerdo,
    run_debate,
)
from providers import BaseProvider


# --- ledger básico ---

def test_record_acuerdo_topea_y_no_duplica():
    r = DebateResult(task="t", mode="plan")
    record_acuerdo(r, "a")
    record_acuerdo(r, "a")
    assert r.acuerdos == ["a"]
    for i in range(12):
        record_acuerdo(r, f"acuerdo {i}")
    assert len(r.acuerdos) == 8
    assert r.acuerdos[-1] == "acuerdo 11"
    assert "acuerdo 0" not in r.acuerdos


def test_prompt_muestra_solo_ultimos_8():
    tr = [Turn(round=1, model="m", text="x", estado="DEBATIENDO")]
    p = build_turn_prompt("t", "", tr, 5, "plan", acuerdos=[f"a{i}" for i in range(12)])
    assert "ACUERDOS_PREVIOS" in p
    assert "- a11\n" in p and "- a4\n" in p
    assert "- a3\n" not in p


def test_prompt_sin_acuerdos_no_bloque():
    tr = [Turn(round=1, model="m", text="x", estado="DEBATIENDO")]
    assert "ACUERDOS_PREVIOS" not in build_turn_prompt("t", "", tr, 2, "plan")
    assert "ACUERDOS_PREVIOS" not in build_turn_prompt("t", "", tr, 2, "plan", acuerdos=[])


# --- peer: la respuesta humana sobrevive a la poda ---

class AskOnce(BaseProvider):
    def __init__(self):
        super().__init__("qa")
        self.n = 0

    def chat(self, messages):
        self.n += 1
        if self.n == 1:
            return "Singleton o dispose? PREGUNTA: cuál?\nESTADO: PREGUNTA_AL_USUARIO"
        return "Ok.\nESTADO: DEBATIENDO"


class Slow(BaseProvider):
    """Consensúa tarde para forzar poda de rondas intermedias."""

    def __init__(self, mid, target=4):
        super().__init__(mid)
        self.target = target
        self.prompts = []

    def chat(self, messages):
        last = messages[-1].get("content", "")
        self.prompts.append(last)
        m = __import__("re").search(r"\[Ronda\s+(\d+)\]", last)
        rnd = int(m.group(1)) if m else 1
        if rnd >= self.target:
            return "De acuerdo.\nESTADO: CONSENSO_ALCANZADO"
        return "Sigo deliberando.\nESTADO: DEBATIENDO"


def test_peer_human_answer_sobrevive_ventana():
    slow1, slow2 = Slow("s1"), Slow("s2")
    r = run_debate(
        [AskOnce(), slow1, slow2], task="t", max_rounds=4,
        interactive=True, ask_user=lambda m, q: "Singleton.",
        verbose=False,
    )
    assert r.acuerdos, "la respuesta humana debió quedar en el ledger"
    assert any("Singleton" in a for a in r.acuerdos)
    late = [p for p in slow1.prompts if "[Ronda 4]" in p]
    assert late, "debió haber prompts de ronda 4"
    assert "omitidas por ventana" in late[-1], "la poda debió activarse"
    assert "ACUERDOS_PREVIOS" in late[-1]
    assert "Singleton" in late[-1], "la decisión humana podada sigue visible"


# --- lead: objeción resuelta visible en rondas tardías ---

class ScriptAdvisor(BaseProvider):
    def __init__(self, mid, script):
        super().__init__(mid)
        self.script = list(script)
        self.n = 0
        self.prompts = []

    def chat(self, messages):
        self.n += 1
        self.prompts.append(messages[-1].get("content", ""))
        return self.script[min(self.n - 1, len(self.script) - 1)]


class PlainLead(BaseProvider):
    def __init__(self):
        super().__init__("L")
        self.prompts = []

    def chat(self, messages):
        last = messages[-1].get("content", "")
        self.prompts.append(last)
        if "diff unificado" in last.lower():
            return "DIFF"
        return "Propuesta.\nESTADO: DEBATIENDO"


OBJECT = "Falta mutex en línea 40.\nESTADO: OBJECION_BLOQUEANTE"
AGREE = "Todo bien.\nESTADO: CONFORME"
CHAT = "Detalle menor.\nESTADO: DEBATIENDO"


def test_lead_objecion_resuelta_no_se_pierde():
    a1 = ScriptAdvisor("a1", [OBJECT, AGREE, AGREE])
    a2 = ScriptAdvisor("a2", [CHAT, CHAT, AGREE])
    lead = PlainLead()
    r = run_lead_debate(lead, [a1, a2], task="t", mode="plan", verbose=False)
    assert r.rounds_used == 3
    assert any("mutex" in a and "no reabrir" in a for a in r.acuerdos), r.acuerdos
    # El líder la vio pendiente en R2 y saldada en R3
    assert any("Objeciones bloqueantes" in p and "mutex" in p for p in lead.prompts)
    assert any("ACUERDOS_PREVIOS" in p and "mutex" in p for p in lead.prompts)
    # El asesor, que es stateless, la recibe reinyectada en R3
    r3 = [p for p in a2.prompts if "[Ronda 3]" in p]
    assert r3 and "ACUERDOS_PREVIOS" in r3[-1] and "mutex" in r3[-1]


def test_lead_veto_queda_registrado():
    veto = "Premisa inválida.\nESTADO: VETO_ARQUITECTONICO"
    a1 = ScriptAdvisor("a1", [veto, AGREE])
    a2 = ScriptAdvisor("a2", [veto, AGREE])
    r = run_lead_debate(PlainLead(), [a1, a2], task="t", mode="plan", verbose=False)
    assert any("veto" in a.lower() for a in r.acuerdos)
