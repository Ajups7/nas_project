"""
Reproduces the scenario table in faoc_loss_validation.md. Not a unit test
suite (no assertions) - a diagnostic script that prints FAOCLoss's behavior
against plain BCE across confusion-matrix corners, airport multipliers, and
confidence levels, so the numbers in the writeup can be regenerated instead
of trusted blind.

Run from the repo root:
    python -m losses.validate_faoc_loss
"""

import torch

from losses.faoc_loss import FAOCLoss

criterion = FAOCLoss()
plain_bce = torch.nn.BCELoss(reduction="none")


def run(name: str, p: float, y: int, airport: str) -> None:
    p_t = torch.tensor([p])
    y_t = torch.tensor([float(y)])
    faoc = criterion(p_t, y_t, [airport]).item()
    plain = plain_bce(p_t.clamp(1e-7, 1 - 1e-7), y_t).item()
    dm = criterion.multipliers.get(airport, 1.51)
    print(
        f"{name:38s} p={p:<5} y={y}  airport={airport:4s} DM={dm:.2f}  "
        f"FAOC={faoc:10,.1f}  plainBCE={plain:6.3f}  ratio={faoc / plain:10,.1f}x"
    )


if __name__ == "__main__":
    print("=== Confusion-matrix corners, fixed airport (ORD) ===")
    run("True negative (confident)", 0.02, 0, "ORD")
    run("False positive (confident)", 0.98, 0, "ORD")
    run("False negative (confident)", 0.02, 1, "ORD")
    run("True positive (confident)", 0.98, 1, "ORD")

    print()
    print("=== Same false negative, varying airport multiplier ===")
    run("FN at LGB (highest, 2.00)", 0.02, 1, "LGB")
    run("FN at ORD (mid, ~1.48)", 0.02, 1, "ORD")
    run("FN at ADK (lowest, 1.19)", 0.02, 1, "ADK")
    run("FN at ZZZ (no entry->1.51)", 0.02, 1, "ZZZ")

    print()
    print("=== Confidence sweep, false negative at ORD ===")
    for p in [0.5, 0.3, 0.1, 0.05, 0.01]:
        run(f"y=1, pred={p}", p, 1, "ORD")

    print()
    print("=== Confidence sweep, false positive at ORD ===")
    for p in [0.5, 0.7, 0.9, 0.95, 0.99]:
        run(f"y=0, pred={p}", p, 0, "ORD")
