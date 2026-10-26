"""
Why does TFT (a baseline) have lower decision cost than PlanningDisruptNet
on validation ($656 vs $714 per airport-day, cost-ratio threshold)?
Uses the saved checkpoints (3 seeds each); decomposes the gap by
error type, by airport, and by calibration near the ~7% decision threshold.

Run from the repo root:
    python -m models.planning.investigate_tft_cost
"""

import numpy as np
import pandas as pd
import torch

from losses.faoc_loss import NATIONAL_COMPOSITE_MULTIPLIER, FAOCLoss
from models.planning.baselines import TFTBaseline
from models.planning.compare_losses import predict_from_checkpoint
from models.planning.metrics import decision_costs
from models.planning.sequences import SEQUENCE_COLUMNS, build_sequence_splits, standardize
from models.planning.train_disruptnet import CHECKPOINT_DIR, RESULTS_DIR

SEEDS = (0, 1, 2)
BINS = [0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.25, 0.5, 0.75, 1.0001]


def tft_predictions(seq_val) -> np.ndarray:
    X = torch.from_numpy(seq_val[0])
    out = []
    for seed in SEEDS:
        model = TFTBaseline(len(SEQUENCE_COLUMNS))
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"baseline_tft_seed{seed}.pt")["state_dict"])
        model.eval()
        with torch.no_grad():
            out.append(torch.cat([model(X[i:i + 4096]) for i in range(0, len(X), 4096)]).numpy())
    return np.mean(out, axis=0)


def main():
    seq = standardize(build_sequence_splits())
    _, y, airports, val = seq["validation"]
    p_tft = tft_predictions(seq["validation"])
    p_dn = np.mean([predict_from_checkpoint(CHECKPOINT_DIR / f"standard_pw1_all_seed{s}.pt", val)
                    for s in SEEDS], axis=0)

    faoc = FAOCLoss()
    dm = np.array([faoc.multipliers.get(a, NATIONAL_COMPOSITE_MULTIPLIER) for a in airports])
    thr = faoc.fp_weight / (faoc.fp_weight + faoc.fn_weight * dm)
    miss_w = faoc.fn_weight * dm

    df = pd.DataFrame({"airport": airports, "date": val["date"].to_numpy(), "y": y,
                       "p_dn": p_dn, "p_tft": p_tft, "thr": thr, "miss_w": miss_w})
    for m in ("dn", "tft"):
        flag = df[f"p_{m}"] >= df["thr"]
        df[f"miss_{m}"] = np.where((df["y"] == 1) & ~flag, df["miss_w"], 0.0)
        df[f"fa_{m}"] = np.where((df["y"] == 0) & flag, faoc.fp_weight, 0.0)
        df[f"flag_{m}"] = flag

    n = len(df)
    print("=== 1. Seed-averaged ensemble cost (cost-ratio threshold) ===")
    for m, name in (("dn", "DisruptNet"), ("tft", "TFT")):
        c = decision_costs(df["y"], df[f"flag_{m}"], df["airport"], faoc)
        print(f"  {name:10s} ${c['cost_per_airport_day']:.0f}/airport-day  flag rate {c['flag_rate']:.3f}  "
              f"miss ${df[f'miss_{m}'].sum() / n:.0f}  false alarm ${df[f'fa_{m}'].sum() / n:.0f}")
    gap = (df["miss_dn"] + df["fa_dn"] - df["miss_tft"] - df["fa_tft"]).sum() / n
    print(f"  gap DisruptNet - TFT: ${gap:.0f}  (miss part ${(df['miss_dn'] - df['miss_tft']).sum() / n:.0f}, "
          f"false-alarm part ${(df['fa_dn'] - df['fa_tft']).sum() / n:.0f})")

    print("\n=== 2. Where the decisions differ ===")
    only_dn = df["flag_dn"] & ~df["flag_tft"]
    only_tft = df["flag_tft"] & ~df["flag_dn"]
    for mask, name in ((only_dn, "flagged by DisruptNet only"), (only_tft, "flagged by TFT only")):
        sub = df[mask]
        print(f"  {name:28s} {len(sub):6,} rows  actually GDP: {sub['y'].mean():.3f}  "
              f"median p_dn {sub['p_dn'].median():.3f}  median p_tft {sub['p_tft'].median():.3f}")

    print("\n=== 3. Gap by airport (top 10 contributors, $ per validation airport-day overall) ===")
    by_ap = df.groupby("airport").apply(lambda g: pd.Series({
        "gap": (g["miss_dn"] + g["fa_dn"] - g["miss_tft"] - g["fa_tft"]).sum() / n,
        "prevalence": g["y"].mean(),
        "flag_dn": g["flag_dn"].mean(),
        "flag_tft": g["flag_tft"].mean(),
    }), include_groups=False).sort_values("gap", ascending=False)
    print(by_ap.head(10).round(3).to_string())
    top = by_ap["gap"].clip(lower=0)
    print(f"  airports with gap > 0: {(by_ap['gap'] > 0).sum()} of {len(by_ap)};  "
          f"top 10 share of total gap: {by_ap['gap'].head(10).sum() / by_ap['gap'].sum():.0%}")
    print("  gap by airport prevalence group:")
    grp = pd.cut(by_ap["prevalence"], [-0.01, 0.05, 0.3, 0.7, 1.01], labels=["quiet <5%", "5-30%", "30-70%", "busy >70%"])
    print(by_ap.groupby(grp, observed=True)["gap"].agg(["sum", "count"]).round(1).to_string())

    print("\n=== 4. Calibration near the decision threshold (~6-9%) ===")
    rows = []
    for m in ("dn", "tft"):
        b = pd.cut(df[f"p_{m}"], BINS, right=False)
        g = df.groupby(b, observed=True).agg(n=("y", "size"), mean_p=(f"p_{m}", "mean"), actual=("y", "mean"))
        g.insert(0, "model", m)
        rows.append(g)
    print(pd.concat(rows).round(3).to_string())

    print("\n=== 5. Do the extra TFT inputs explain it? Daily GDP history at quiet-looking rows ===")
    low = df[(df["p_dn"] >= df["thr"]) & (df["p_tft"] < df["thr"])]
    hist = seq["validation"][0][low.index.to_numpy(), :, -1] if len(low) else np.empty((0, 30))
    raw = val.loc[low.index, "recent_gdp_rate_30d"]
    print(f"  rows DisruptNet flags but TFT doesn't: {len(low):,};  their recent 30-day GDP rate: "
          f"mean {raw.mean():.3f}, share exactly 0: {(raw == 0).mean():.3f}")

    out = RESULTS_DIR / "tft_vs_disruptnet_validation_rows.parquet"
    df.to_parquet(out, index=False)
    print(f"\nrow-level comparison saved to {out}")


if __name__ == "__main__":
    main()
