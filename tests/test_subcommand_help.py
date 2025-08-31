import subprocess
from pathlib import Path
import pytest

def completion_available(cmd: str) -> bool:
    script = (
        f"[[ -f /usr/share/bash-completion/bash_completion ]] && "
        f"source /usr/share/bash-completion/bash_completion; "
        f"compfile=/usr/share/bash-completion/completions/{cmd}; "
        f"[[ -f $compfile ]] && source $compfile; "
        f"complete -p {cmd} >/dev/null 2>&1"
    )
    return subprocess.run(["bash", "-lc", script], check=False).returncode == 0

def ensure_pip_completion():
    if not completion_available("pip"):
        subprocess.run(
            ["bash", "-lc", "pip completion --bash > /usr/share/bash-completion/completions/pip"],
            check=True,
        )
        assert completion_available("pip")


def test_subcommand_help_generates_files(tmp_path):
    ensure_pip_completion()
    script = Path(__file__).resolve().parent.parent / "subcommand_help.sh"
    outdir = tmp_path / "pip_help"
    subprocess.run([str(script), "pip", str(outdir), "1"], check=True)
    install_file = outdir / "pip_install_help.txt"
    assert install_file.exists()
    assert "pip install" in install_file.read_text()
