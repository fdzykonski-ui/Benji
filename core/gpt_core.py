"""
F-Bot · gpt_core.py
Leichtgewichtiges Interface-Modul ("Code-X") zwischen Chat (Anweisung) und Repo (Änderung).
Kapselt Operationen wie create_module/ensure_file und gibt bewusst wenig vor.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Change:
    path: Path
    content: str
    mode: str = "write"   # "write" | "append"
    encoding: str = "utf-8"


class GPTCore:
    """
    Minimaler Agent-Adapter:
    - nimmt bereits entschiedene Änderungen als 'Change' entgegen
    - schreibt Dateien (Commit/Push liegt außerhalb – Git-Gateway oder Codex übernimmt)
    """

    def __init__(self, repo_root: str | Path = ".") -> None:
        self.root = Path(repo_root).resolve()

    def _write(self, change: Change) -> Path:
        p = (self.root / change.path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if change.mode == "append" and p.exists():
            old = p.read_text(encoding=change.encoding)
            if not old.endswith("\n"):
                old += "\n"
            p.write_text(old + change.content, encoding=change.encoding)
        else:
            p.write_text(change.content, encoding=change.encoding)
        return p

    def apply(self, changes: Iterable[Change]) -> list[Path]:
        return [self._write(ch) for ch in changes]

    # Convenience
    def create_module(self, relpath: str, code: str) -> Path:
        return self._write(Change(path=Path(relpath), content=code, mode="write"))

    def ensure_file(self, relpath: str, content: str = "") -> Path:
        p = self.root / relpath
        if not p.exists():
            self.create_module(relpath, content)
        return p


if __name__ == "__main__":
    core = GPTCore()
    core.ensure_file("README.md", "# F-Bot\n")
    print("gpt_core ready →", core.root)
