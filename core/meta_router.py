"""
F-Bot · meta_router.py
Meta-Router für Strategien/Backtests/Hyperopt. Hält nur die "Ports" – die Ausführung
passiert in Notebook/CLI. Ziel: dünnes, sauberes, testbares API.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any


@dataclass
class StrategyRef:
    name: str
    file: Path
    config: Path | None = None


@dataclass
class RunPlan:
    pairs: list[str]
    timeframes: list[str]      # e.g. ['5m','15m','1h','4h']
    timerange: str             # e.g. '20240101-'
    smoke: bool = False


class MetaRouter:
    def __init__(self, userdir: str | Path = "hard_user_data") -> None:
        self.userdir = Path(userdir)
        self.strategies_dir = self.userdir / "strategies"
        self.configs_dir = self.userdir
        self.results_dir = self.userdir / "hyperopt_results"
        self.validate_dir = self.userdir / "validate_results"
        self.logs_dir = self.userdir / "logs"

    def discover_strategies(self) -> list[StrategyRef]:
        out: list[StrategyRef] = []
        if not self.strategies_dir.exists():
            return out
        for py in self.strategies_dir.glob("*.py"):
            # Heuristik: Klassen-/Dateiname endet auf 'Strategy'
            stem = py.stem
            name = stem if stem.lower().endswith("strategy") else stem
            cfg = next((c for c in self.configs_dir.glob(f"config*{name}*.json")), None)
            out.append(StrategyRef(name=name, file=py, config=cfg))
        return sorted(out, key=lambda s: s.name.lower())

    def plan_backtests(self, plan: RunPlan) -> list[Mapping[str, Any]]:
        runs: list[Mapping[str, Any]] = []
        for sref in self.discover_strategies():
            for tf in plan.timeframes:
                runs.append({
                    "kind": "backtest",
                    "strategy": sref.name,
                    "file": str(sref.file),
                    "config": str(sref.config) if sref.config else None,
                    "pairs": plan.pairs,
                    "timeframe": tf,
                    "timerange": plan.timerange,
                    "smoke": plan.smoke,
                })
        return runs

    def plan_hyperopt(self, plan: RunPlan, spaces: str = "default", epochs: int = 200) -> list[Mapping[str, Any]]:
        runs: list[Mapping[str, Any]] = []
        for sref in self.discover_strategies():
            for tf in plan.timeframes:
                runs.append({
                    "kind": "hyperopt",
                    "strategy": sref.name,
                    "file": str(sref.file),
                    "config": str(sref.config) if sref.config else None,
                    "pairs": plan.pairs,
                    "timeframe": tf,
                    "timerange": plan.timerange,
                    "spaces": spaces,
                    "epochs": epochs,
                })
        return runs

    def report_paths(self) -> dict[str, str]:
        return {"results": str(self.results_dir),
                "validate": str(self.validate_dir),
                "logs": str(self.logs_dir)}


if __name__ == "__main__":
    mr = MetaRouter()
    print("strategies:", [s.name for s in mr.discover_strategies()])
    print("paths:", mr.report_paths())
