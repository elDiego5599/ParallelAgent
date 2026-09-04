"""Macro-cascada condicional: fast-exit, expansión, ping-pong y consentimiento."""

import subprocess

import pytest

from cli import build_parser, validate_and_infer_topology
from git_bridge import current_branch
from macro_engine import (
    CycleBundle,
    MacroEngine,
    PingPongDetector,
    parse_audit_estado,
)


DIFF1 = """diff --git a/f.txt b/f.txt
--- a/f.txt
+++ b/f.txt
@@ -1 +1,2 @@
 hola
+linea ciclo 1
"""

DIFF2 = """diff --git a/f.txt b/f.txt
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,3 @@
 hola
 linea ciclo 1
+linea ciclo 2
"""


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def init_repo(path):
    git("init", "-q", cwd=path)
    git("config", "user.email", "t@t.t", cwd=path)
    git("config", "user.name", "t", cwd=path)
    (path / "f.txt").write_text("hola\n")
    git("add", "-A", cwd=path)
    git("-c", "user.name=t", "-c", "user.email=t@t.t", "commit", "-qm", "init", cwd=path)


def branches(path):
    return subprocess.run(
        ["git", "branch", "--list", "consensus/*"],
        cwd=path, capture_output=True, text=True, check=True,
    ).stdout


def log_messages(path, branch):
    r = subprocess.run(
        ["git", "log", branch, "--format=%s"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    return r.stdout


def make_bundle(diff, files=None, audits=(), repairs=None):
    audits = list(audits)
    state = {"n": 0}

    def audit_chat(prompt):
        state["n"] += 1
        if audits:
            return audits.pop(0)
        return "ESTADO: TAREA_FINALIZADA"

    def repair_chat(prompt):
        return repairs if isinstance(repairs, str) else DIFF1

    return CycleBundle(
        result=None, diff=diff, files=files or ["f.txt"],
        writer_label="w", repair_context="ctx",
        audit_chat=audit_chat, repair_chat=repair_chat,
    )


def parse_cli(tmp_path, *argv):
    return build_parser().parse_args(["--task", "t", "--path", str(tmp_path), *argv])


# --- audit parser ---

def test_parse_audit():
    assert parse_audit_estado("todo bien\nESTADO: TAREA_FINALIZADA") == ("TAREA_FINALIZADA", "")
    e, f = parse_audit_estado("ojo\nESTADO: NUEVO_HALLAZGO: falta null en dart")
    assert e == "NUEVO_HALLAZGO" and "null" in f
    # Malformado -> fast-exit seguro, nunca expande por accidente
    assert parse_audit_estado("bla bla sin estado")[0] == "TAREA_FINALIZADA"
    assert parse_audit_estado("")[0] == "TAREA_FINALIZADA"


# --- ping-pong ---

def test_pingpong_detecta_revert():
    d = PingPongDetector()
    d.observe(["lib/a.dart"], "quitar mutex para velocidad en bridge")
    loop, _ = d.is_oscillation(["lib/a.dart"], "quitar mutex velocidad bridge otra vez")
    assert loop is True


def test_pingpong_no_frena_hallazgo_nuevo():
    d = PingPongDetector()
    d.observe(["a.cpp"], "fuga de memoria en puntero nativo")
    loop, _ = d.is_oscillation(["lib/b.dart"], "null en methodchannel de dart")
    assert loop is False


# --- fast-exit ---

def test_fast_exit_un_ciclo(tmp_path):
    init_repo(tmp_path)
    calls = []

    def cycle_fn(task):
        calls.append(task)
        return make_bundle(DIFF1)

    rc, res = MacroEngine(
        tmp_path, "fix simple", max_cycles=3, cycle_fn=cycle_fn,
        interactive=False, auto_approve=True, verbose=False,
    ).run()
    assert rc == 0 and res.commits == 1 and res.cycles_used == 1
    assert res.finished is True and res.findings == []
    assert "consensus/" in branches(tmp_path)
    assert "consensus(c1)" in log_messages(tmp_path, res.branch)
    assert current_branch(tmp_path) != res.branch  # volvió al origen


# --- expansión bajo demanda ---

def test_expansion_dos_ciclos(tmp_path):
    init_repo(tmp_path)
    diffs = [DIFF1, DIFF2]
    audits = ["ESTADO: NUEVO_HALLAZGO: falta segunda línea", "ESTADO: TAREA_FINALIZADA"]

    def cycle_fn(task):
        d = diffs.pop(0)
        a = [audits.pop(0)] if audits else ["ESTADO: TAREA_FINALIZADA"]
        return make_bundle(d, audits=a)

    rc, res = MacroEngine(
        tmp_path, "fix cascada", max_cycles=3, cycle_fn=cycle_fn,
        interactive=False, auto_approve=True, verbose=False,
    ).run()
    assert rc == 0 and res.commits == 2 and res.cycles_used == 2
    assert res.findings == ["falta segunda línea"]
    log = log_messages(tmp_path, res.branch)
    assert "consensus(c1)" in log and "consensus(c2)" in log


def test_techo_max_cycles(tmp_path):
    init_repo(tmp_path)
    calls = {"n": 0}

    def cycle_fn(task):
        calls["n"] += 1
        d = DIFF1 if calls["n"] == 1 else DIFF2
        return make_bundle(d, audits=["ESTADO: NUEVO_HALLAZGO: siempre hay algo más x%d" % calls["n"]])

    rc, res = MacroEngine(
        tmp_path, "t", max_cycles=2, cycle_fn=cycle_fn,
        interactive=False, auto_approve=True, verbose=False,
    ).run()
    assert res.commits == 2 and res.cycles_used == 2
    assert "max-cycles" in res.stopped_reason


# --- consentimiento: sin commit silencioso ---

def test_commit_declinado_no_commitea_y_limpia_rama(tmp_path):
    init_repo(tmp_path)

    def cycle_fn(task):
        return make_bundle(DIFF1)

    rc, res = MacroEngine(
        tmp_path, "t", max_cycles=3, cycle_fn=cycle_fn,
        interactive=True, auto_approve=False, verbose=False,
        ask_commit=lambda c, s, b, f: False,
    ).run()
    assert rc == 0 and res.commits == 0
    assert res.branch == "" and "consensus/" not in branches(tmp_path)
    assert (tmp_path / "f.txt").read_text() == "hola\n"


def test_continue_declinado_conserva_ciclo1(tmp_path):
    init_repo(tmp_path)

    def cycle_fn(task):
        return make_bundle(DIFF1, audits=["ESTADO: NUEVO_HALLAZGO: algo más"])

    rc, res = MacroEngine(
        tmp_path, "t", max_cycles=3, cycle_fn=cycle_fn,
        interactive=True, auto_approve=False, verbose=False,
        ask_commit=lambda c, s, b, f: True,
        ask_continue=lambda c, f: False,
    ).run()
    assert res.commits == 1 and "frenó" in res.stopped_reason


def test_push_solo_con_flag(tmp_path, monkeypatch):
    import macro_engine as me

    init_repo(tmp_path)
    pushed = []
    monkeypatch.setattr(me, "push_branch", lambda p, b, remote="origin": pushed.append(b) or "ok")

    def cycle_fn(task):
        return make_bundle(DIFF1)

    _, res = MacroEngine(
        tmp_path, "t", max_cycles=2, cycle_fn=cycle_fn,
        interactive=False, auto_approve=True, push=False, verbose=False,
    ).run()
    assert pushed == [] and res.pushed is False

    _, res2 = MacroEngine(
        tmp_path, "t2", max_cycles=2, cycle_fn=cycle_fn,
        interactive=False, auto_approve=True, push=True, verbose=False,
    ).run()
    assert len(pushed) == 1 and res2.pushed is True


def test_fallo_apply_preserva_ciclos_previos(tmp_path):
    init_repo(tmp_path)
    seq = [DIFF1, "basura no diff"]

    def cycle_fn(task):
        return make_bundle(seq.pop(0), audits=["ESTADO: TAREA_FINALIZADA"],
                            repairs="todavía basura")

    rc, res = MacroEngine(
        tmp_path, "t", max_cycles=3, cycle_fn=cycle_fn,
        interactive=False, auto_approve=True, verbose=False,
    ).run()
    assert rc == 0 and res.commits == 1  # c1 intacto, c2 falló sin borrar rama
    assert "consensus/" in branches(tmp_path)


# --- CLI ---

def test_cli_max_cycles_flags(tmp_path):
    args = parse_cli(tmp_path, "--models", "a", "b", "--max-cycles", "3", "--yes", "--push")
    assert args.max_cycles == 3 and args.yes is True and args.push is True
    assert validate_and_infer_topology(args) == "peer"


def test_cli_push_solo_build(tmp_path):
    with pytest.raises(ValueError, match="--push solo"):
        validate_and_infer_topology(
            parse_cli(tmp_path, "--models", "a", "b", "--mode", "plan", "--push")
        )


def test_cli_cascada_solo_build(tmp_path):
    with pytest.raises(ValueError, match="--max-cycles"):
        validate_and_infer_topology(
            parse_cli(tmp_path, "--models", "a", "b", "--mode", "ask", "--max-cycles", "3")
        )
