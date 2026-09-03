"""Contexto: ranking, presupuesto, exclusiones y fallback sin git."""

from context import (
    build_repo_context,
    build_repo_tree,
    extract_keywords,
    get_repo_files,
    is_binary_file,
    score_file,
)


def test_keywords_drop_stopwords():
    kw = extract_keywords("Fix memory leak en el bridge JNI para Android")
    assert {"leak", "bridge", "jni", "memory", "android"} <= kw
    assert "fix" not in kw and "para" not in kw


def test_path_match_outranks_content():
    high = score_file("native/bridge.cpp", "x", {"bridge"})
    low = score_file("docs/notas.md", "bridge " * 20, {"bridge"})
    assert high > low


def test_tree_truncates():
    from pathlib import Path

    files = [Path(f"r/f{i}.py") for i in range(50)]
    tree = build_repo_tree(files, Path("r"), max_lines=10)
    assert "40 archivos más" in tree


def test_build_ranks_and_respects_budget(tmp_path):
    (tmp_path / "bridge.cpp").write_text("leak en el bridge nativo\n" * 50)
    (tmp_path / "notas.md").write_text("receta de cocina\n")
    (tmp_path / "package-lock.json").write_text('{"bridge": true}')
    (tmp_path / "bin.py").write_bytes(b"\x00\x01binario")
    ctx = build_repo_context(tmp_path, "fuga leak en bridge", max_total_chars=30000)
    assert "MAPA DE CONTEXTO" in ctx
    assert ctx.index("bridge.cpp") < ctx.index("notas.md")
    assert "[ARCHIVO: package-lock.json]" not in ctx
    assert '{"bridge": true}' not in ctx
    assert "[ARCHIVO: bin.py]" not in ctx


def test_budget_is_exact(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n" * 2000)
    for budget in (30000, 5000, 2000):
        ctx = build_repo_context(tmp_path, "t", max_total_chars=budget)
        assert len(ctx) <= budget, (budget, len(ctx))


def test_fallback_without_git(tmp_path):
    (tmp_path / "a.py").write_text("x=1")
    assert any(f.name == "a.py" for f in get_repo_files(tmp_path))
    assert not is_binary_file(tmp_path / "a.py")
