"""Vector 1: emisión multi-archivo en 2 pasos (Declarar -> Reinyectar -> Diff)."""

import subprocess
from pathlib import Path

from lead_engine import run_lead_debate
from orchestrator import (
    _parse_file_list,
    ask_files_to_modify,
    build_emission_prompt,
    build_multi_file_base_context,
    build_repair_prompt,
    run_debate,
)
from git_bridge import apply_consensus_to_branch, current_branch
from providers import BaseProvider


DART = "lib/bridge.dart"
KT = "android/app/src/main/kotlin/MainActivity.kt"

DART_ORIG = "class Bridge {}\n"
KT_ORIG = "class MainActivity\n"

DIFF_TWO = """diff --git a/lib/bridge.dart b/lib/bridge.dart
--- a/lib/bridge.dart
+++ b/lib/bridge.dart
@@ -1 +1,2 @@
 class Bridge {}
+// patched dart
diff --git a/android/app/src/main/kotlin/MainActivity.kt b/android/app/src/main/kotlin/MainActivity.kt
--- a/android/app/src/main/kotlin/MainActivity.kt
+++ b/android/app/src/main/kotlin/MainActivity.kt
@@ -1 +1,2 @@
 class MainActivity
+// patched kotlin
"""

FILES_TEXT = f"{DART}\n{KT}\n"


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def init_two_file_repo(path: Path):
    git("init", "-q", cwd=path)
    git("config", "user.email", "t@t.t", cwd=path)
    git("config", "user.name", "t", cwd=path)
    (path / "lib").mkdir(parents=True, exist_ok=True)
    (path / "android/app/src/main/kotlin").mkdir(parents=True, exist_ok=True)
    (path / DART).write_text(DART_ORIG)
    (path / KT).write_text(KT_ORIG)
    git("add", "-A", cwd=path)
    git("-c", "user.name=t", "-c", "user.email=t@t.t", "commit", "-qm", "init", cwd=path)


def show(path: Path, branch: str, name: str) -> str:
    r = subprocess.run(
        ["git", "show", f"{branch}:{name}"],
        cwd=path, capture_output=True, text=True, check=True,
    )
    return r.stdout


class WriterPeer(BaseProvider):
    def __init__(self, mid, files_text=FILES_TEXT, diff=DIFF_TWO):
        super().__init__(mid)
        self.files_text = files_text
        self.diff = diff
        self.saw_declaration = False
        self.saw_emission = False
        self.last_emission = ""

    def chat(self, messages):
        last = messages[-1].get("content", "")
        low = last.lower()
        if "antes de generar el diff" in low:
            self.saw_declaration = True
            return self.files_text
        if "diff unificado" in low:
            self.saw_emission = True
            self.last_emission = last
            return self.diff
        return "De acuerdo con la mesa.\nESTADO: CONSENSO_ALCANZADO"


class SimplePeer(BaseProvider):
    def chat(self, messages):
        return "Ok, coincido.\nESTADO: CONSENSO_ALCANZADO"


class ScriptedLead(BaseProvider):
    def __init__(self, mid="L", files_text=FILES_TEXT, diff=DIFF_TWO):
        super().__init__(mid)
        self.files_text = files_text
        self.diff = diff
        self.last_emission = ""

    def chat(self, messages):
        last = messages[-1].get("content", "")
        low = last.lower()
        if "antes de generar el diff" in low:
            return self.files_text
        if "diff unificado" in low:
            self.last_emission = last
            return self.diff
        return "Propuesta base.\nESTADO: DEBATIENDO"


class ConformeAdvisor(BaseProvider):
    def chat(self, messages):
        return "LGTM.\nESTADO: CONFORME"


# --- parsing ---

def test_parse_file_list_tolera_viñetas_y_filtra_basura():
    resp = (
        "- lib/bridge.dart\n"
        "* `android/app/src/main/kotlin/MainActivity.kt`\n"
        "1. lib/new.dart\n"
        "Creo que eso es todo.\n"
        "# comentario\n"
        "```\n"
    )
    out = _parse_file_list(resp)
    assert out == [
        "lib/bridge.dart",
        "android/app/src/main/kotlin/MainActivity.kt",
        "lib/new.dart",
    ]


def test_parse_file_list_rechaza_traversal_y_absolutas():
    resp = "/etc/passwd\n../fuera.txt\nlib/ok.dart\n"
    assert _parse_file_list(resp) == ["lib/ok.dart"]


def test_parse_file_list_tope_y_dedup():
    resp = "lib/a.dart\nlib/a.dart\nsin ruta con espacios\n"
    assert _parse_file_list(resp) == ["lib/a.dart"]


# --- base context ---

def test_base_context_existente_y_nuevo(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/a.dart").write_text("hola\n")
    out = build_multi_file_base_context(tmp_path, ["lib/a.dart", "lib/nuevo.dart"])
    assert "ARCHIVO EXISTENTE: lib/a.dart" in out
    assert "hola" in out
    assert "ARCHIVO NUEVO A CREAR: lib/nuevo.dart" in out


def test_base_context_omite_fuera_del_repo(tmp_path):
    out = build_multi_file_base_context(tmp_path, ["../evil.txt", "lib/a.dart"])
    assert "OMITIDA" in out


def test_ask_files_fallback_ante_proveedor_roto():
    class Boom(BaseProvider):
        def chat(self, messages):
            raise AssertionError("no debería llamarse con prompt viejo")
            # ask_files captura ProviderError, no AssertionError; lo envolvemos:
    from providers import ProviderError

    class Fail(BaseProvider):
        def chat(self, messages):
            raise ProviderError("caído")

    assert ask_files_to_modify(Fail("w"), "tarea") == []


# --- end to end peer ---

def test_peer_two_file_patch_end_to_end(tmp_path):
    init_two_file_repo(tmp_path)
    writer = WriterPeer("w")
    r = run_debate(
        [writer, SimplePeer("p2")],
        task="arreglar bridge",
        mode="build",
        max_rounds=2,
        writer="w",
        verbose=False,
        project_path=tmp_path,
    )
    assert r.files_declared == [DART, KT]
    assert "lib/bridge.dart" in r.base_context
    assert "MainActivity" in r.base_context
    assert writer.saw_declaration and writer.saw_emission
    # La base exacta fue reinyectada: el redactor no alucina hunks
    assert "class Bridge" in writer.last_emission
    assert "class MainActivity" in writer.last_emission
    assert "--- a/lib/bridge.dart" in r.final_output
    assert "--- a/android/app/src/main/kotlin/MainActivity.kt" in r.final_output

    base = current_branch(tmp_path)
    info = apply_consensus_to_branch(tmp_path, "fix bridge", r.final_output)
    assert show(tmp_path, info["branch"], DART) == DART_ORIG + "// patched dart\n"
    assert show(tmp_path, info["branch"], KT) == KT_ORIG + "// patched kotlin\n"
    assert current_branch(tmp_path) == base


def test_peer_fallback_si_declara_basura(tmp_path):
    init_two_file_repo(tmp_path)
    writer = WriterPeer("w", files_text="no sé, lo que sea\n")
    # Declaración sin forma de ruta -> fallback a mono-contexto

    def patched(messages):
        last = messages[-1].get("content", "")
        low = last.lower()
        if "antes de generar el diff" in low:
            return "bla bla sin forma de ruta con espacios"
        if "diff unificado" in low:
            writer.saw_emission = True
            return DIFF_TWO
        return "Ok.\nESTADO: CONSENSO_ALCANZADO"

    writer.chat = patched
    r = run_debate(
        [writer, SimplePeer("p2")], task="t", mode="build",
        max_rounds=2, writer="w", verbose=False, project_path=tmp_path,
    )
    assert r.files_declared == []
    assert r.base_context == ""
    assert "--- a/lib/bridge.dart" in r.final_output


def test_peer_sin_project_path_no_declara():
    w = WriterPeer("w")
    r = run_debate(
        [w, SimplePeer("p2")], task="t", mode="build",
        max_rounds=1, writer="w", verbose=False, project_path=None,
    )
    assert r.files_declared == []
    assert w.saw_declaration is False


# --- end to end lead ---

def test_lead_two_file_patch_end_to_end(tmp_path):
    init_two_file_repo(tmp_path)
    lead = ScriptedLead()
    r = run_lead_debate(
        lead, [ConformeAdvisor("a1")],
        task="arreglar bridge", mode="build",
        verbose=False, project_path=tmp_path,
    )
    assert r.files_declared == [DART, KT]
    assert "class Bridge" in lead.last_emission
    assert "--- a/lib/bridge.dart" in r.final_output
    info = apply_consensus_to_branch(tmp_path, "fix", r.final_output)
    assert show(tmp_path, info["branch"], DART).endswith("// patched dart\n")


def test_repair_prompt_reinyecta_base_multifile(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/a.dart").write_text("contenido exacto\n")
    base = build_multi_file_base_context(tmp_path, ["lib/a.dart"])
    prompt = build_repair_prompt("t", "error en hunk 2 de lib/a.dart", base)
    assert "contenido exacto" in prompt
    assert "error en hunk" in prompt


def test_emission_prompt_incluye_bloque_base():
    from orchestrator import Turn
    tr = [Turn(round=1, model="m", text="acuerdo", estado="CONSENSO_ALCANZADO")]
    p = build_emission_prompt("t", tr, "build", context="ctx", base_files="BASE_EXACTA")
    assert "BASE_EXACTA" in p
    assert "--- a/<ruta>" in p
    assert "diff --git" in p
    # Compat: llamada vieja sin base sigue funcionando
    p2 = build_emission_prompt("t", tr, "build", "ctx")
    assert "BASE_EXACTA" not in p2


def test_legacy_multifile_sin_diff_git_tambien_aplica(tmp_path):
    """Compat: LLM viejo sin cabeceras diff --git sigue aplicando vía fallback."""
    init_two_file_repo(tmp_path)
    legacy = (
        "--- a/lib/bridge.dart\n+++ b/lib/bridge.dart\n"
        "@@ -1 +1,2 @@\n class Bridge {}\n+// legacy\n"
        "--- a/android/app/src/main/kotlin/MainActivity.kt\n"
        "+++ b/android/app/src/main/kotlin/MainActivity.kt\n"
        "@@ -1 +1,2 @@\n class MainActivity\n+// legacy\n"
    )
    info = apply_consensus_to_branch(tmp_path, "legacy", legacy)
    assert show(tmp_path, info["branch"], DART).endswith("// legacy\n")
