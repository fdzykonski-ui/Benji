# F-Bot

Leichtgewichtiges Core-Repo für Strategie-Orchestrierung (Freqtrade/Backtests/Hyperopt) + KI-Layer.

## Module
- `core/gpt_core.py` – Code‑X/Agent‑Adapter (Dateierzeugung/-änderung)
- `core/meta_router.py` – Router (Discovery, Run‑Pläne für Backtests/Hyperopt)
- `core/azure_brain.py` – Scoring/Analytics (erweiterbar)

## Quickstart
1) `pip install -r requirements.txt`
2) Lege Strategien in `strategies/`, Configs in `configs/` ab.
3) Nutze Notebook/CLI, um Backtests/Hyperopt zu fahren – oder binde `MetaRouter` direkt ein.
