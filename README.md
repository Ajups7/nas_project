# NAS-DisruptNet

A hierarchical deep learning framework for probabilistic, multi-horizon
forecasting of disruptive regime transitions (Ground Delay Programs, Ground
Stops, Miles-in-Trail restrictions) in the U.S. National Airspace System —
using loss functions, architectures, and evaluation criteria aligned with
FAA/ATCSCC operational decision-making rather than generic statistical
accuracy.

Full goal, objectives, and phased plan: **[PLAN.md](PLAN.md)**

## Status

Phase 1 (data foundation) is mostly done:

| Dataset | Status | Role |
|---|---|---|
| BTS On-Time Performance | ✅ 121/121 months (2016-01 to 2026-01) | flight-level delays, rotation chains |
| ASPM Airport Analysis | ✅ 121/121 months | airport-hour capacity/demand |
| ASPM EDCT Report | ✅ 119/121 months (2 known gaps) | GDP proxy label |
| METAR Weather | ✅ 82/82 airports | observed hourly conditions |
| ASPM Advisories | ⛔ blocked on FAA account approval | real GDP/GS/MIT labels |
| FAOC-Loss cost figures | 📄 sourced, not yet implemented | see PLAN.md |

**Phase 0 (shared schema + data loader) is done and tested** —
[schema.md](schema.md) + [common/data_loader.py](common/data_loader.py).
This is what Phase 2 (per-tier feature engineering and baseline models)
builds on top of.

## Repo layout

```
common/           shared data loader (Phase 0) - load_bts(), load_airport_analysis(), load_edct(), load_weather()
scripts/          download pipelines for all four datasets
data/             downloaded datasets (~13GB, gitignored - lives on the shared VM, see SETUP.md)
schema.md         join keys + column reference for the shared data loader
PLAN.md           full project plan: goal, objectives, phases, data status
TEAM_PLAN.md      how the 3-person team's work is split (Tactical/Strategic/Planning tiers)
SETUP.md          shared VM setup, getting the data, GPU training via RunPod
```

## Getting started

- **New to the project?** Read [PLAN.md](PLAN.md) first, then [TEAM_PLAN.md](TEAM_PLAN.md) for how the work is divided.
- **Setting up your environment?** See [SETUP.md](SETUP.md).
- **Building a tier's feature pipeline?** Start from [schema.md](schema.md) and `common/data_loader.py` — don't re-parse the raw files yourself, the loader already handles the format quirks (see schema.md's "Known airport-code quirks" section).
- **Re-downloading a dataset from scratch?** The four pipelines are in `scripts/` — `download_bts_ontime.py`, `download_aspm_airport_analysis.py` (use `--reptype r1` for Airport Analysis, `--reptype r5` for EDCT), `download_weather_metar.py`.

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
playwright install chromium  # only needed for the ASPM download scripts
```
