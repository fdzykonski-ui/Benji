"""
F-Bot · azure_brain.py
Analytik-/Scoring-Layer. Einfacher, erweiterbarer Score für Resultate (Sharpe, MaxDD, Trades, Profit).
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class Metrics:
    sharpe_tn: float = math.nan
    max_dd: float = math.nan
    trades: int = 0
    profit_total: float = 0.0


class AzureBrain:
    def score(self, m: Metrics) -> float:
        """
        Einfache Heuristik:
        - wenig Trades -> schlechter Score
        - Sharpe hoch -> besser
        - max_dd (negativ) penalisiert
        - Profit skaliert mild
        """
        if m.trades < 10:
            return -1.0
        dd_penalty = 0.0 if (math.isnan(m.max_dd) or m.max_dd >= 0) else min(1.0, abs(m.max_dd))
        sharpe = 0.0 if math.isnan(m.sharpe_tn) else m.sharpe_tn
        base = (0.6 * sharpe) + (0.3 * (m.profit_total / 100.0)) - (0.4 * dd_penalty)
        return float(base)
