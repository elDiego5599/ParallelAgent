import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def init_git_repo(path):
    """Crea un repo git mínimo con un commit para pruebas."""

    def git(*args):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    (path / "f.txt").write_text("hola\n")
    git("add", "-A")
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t.t",
         "commit", "-qm", "init"],
        cwd=path, check=True, capture_output=True,
    )
