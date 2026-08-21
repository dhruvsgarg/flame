"""Parse P4's arm ledger out of `fl_fwd_ft_practice.md`.

The ledger is the source of truth for every historical number (R2: one number,
one home), so the figures read it rather than carrying a second copy that can
drift. Columns: Lambda | arm | run | T | rho_c1 | B | Phi pred -> obs | peak | final
"""
import os
import re

DOC = os.path.join(os.path.dirname(__file__), "..", "..", "fl_fwd_ft_practice.md")

_ROW = re.compile(
    r"^\|\s*([\d.]+)\s*\|"          # Lambda
    r"\s*(.+?)\s*\|"                # arm description
    r"\s*`(\w+)`\s*\|"              # run id
    r"\s*(\d+)\s*\|"                # T (commits)
    r"\s*([\d.]+)\s*\|"             # rho at commit 1
    r"\s*([\d.]+)\s*\|"             # B
    r"\s*([\d.]+)\s*(?:→|->)\s*\*{0,2}([\d.]+)\*{0,2}\s*\|"   # Phi pred -> obs
    r"\s*\*{0,2}([\d.]+)\*{0,2}\s*\|"                          # peak
    r"\s*\*{0,2}([\d.]+)\*{0,2}\s*\|"                          # final
)


def _strip(s):
    return re.sub(r"[`*]", "", s).strip()


def load():
    """Every ledger row as a dict, with the derived flags the figures key on."""
    out = []
    with open(os.path.normpath(DOC), encoding="utf-8") as fh:
        inside = False
        for line in fh:
            if line.startswith("## P4 — Arm ledger"):
                inside = True
                continue
            if inside and line.startswith("### "):
                break
            if not inside:
                continue
            m = _ROW.match(line)
            if not m:
                continue
            lam, desc, run, T, rho1, B, phi_p, phi_o, peak, final = m.groups()
            desc = _strip(desc)
            # `rf` defaults to 16; Lambda does not transfer across p, so the
            # accuracy-vs-Lambda figure must separate these.
            rf = 16
            if "rf=32" in desc.replace("`", "").replace(" ", ""):
                rf = 32
            elif "rf=64" in desc.replace("`", "").replace(" ", ""):
                rf = 64
            out.append(dict(
                lam=float(lam), desc=desc, run=run, T=int(T), rho1=float(rho1),
                B=float(B), phi_pred=float(phi_p), phi_obs=float(phi_o),
                peak=float(peak), final=float(final), rf=rf,
                # Server momentum correlates steps, so Phi follows
                # exp(((1+b)/(1-b))B) -- the norm law's one documented exception.
                beta=("β=0.5" in desc and 0.5) or ("β=0.75" in desc and 0.75) or 0.0,
            ))
    return out


# The 2026-08-20 P-4 pairs (P4.11). Not in the ledger table above, which predates
# them; measured this session from `server_update` telemetry and replay_scoring.
ARMS_2026_08_20 = [
    # dataset,  role,        run,      commits, B,      Phi_pred, Phi_obs, Lambda, peak,   final
    ("agnews", "controller", "152215",  899, 0.6972, 2.01, 2.01, 1.001, 0.8676, 0.8661),
    ("agnews", "control",    "021843",  938, 0.1075, 1.11, 1.11, 0.570, 0.8432, 0.8430),
    ("yahoo",  "controller", "125003",  964, 0.6900, 1.99, 2.00, 0.994, 0.6571, 0.6571),
    ("yahoo",  "control",    "151619", 1138, 0.1187, 1.13, 1.12, 0.655, 0.4275, 0.4158),
    ("yelp-p", "controller", "125010", 1348, 0.9699, 2.64, 2.64, 1.402, 0.8141, 0.8135),
    ("yelp-p", "control",    "161751",  997, 0.1109, 1.12, 1.12, 0.597, 0.7280, 0.7139),
]

# Backprop reference per dataset (exact-gradient, 10 clients x 3 epochs) -- §5.1.
REFERENCE = {"agnews": 0.850, "yahoo": 0.734, "yelp-p": 0.874}


def recent():
    keys = ("dataset", "role", "run", "T", "B", "phi_pred", "phi_obs", "lam",
            "peak", "final")
    return [dict(zip(keys, r)) for r in ARMS_2026_08_20]


if __name__ == "__main__":
    rows = load()
    print(f"parsed {len(rows)} ledger rows")
    for r in rows[:3] + rows[-3:]:
        print(f"  L={r['lam']:.3f} {r['run']} T={r['T']:4d} B={r['B']:.4f} "
              f"Phi {r['phi_pred']}->{r['phi_obs']} peak={r['peak']} "
              f"final={r['final']} rf={r['rf']} beta={r['beta']}")
    print(f"drops > 0.05: {sum(1 for r in rows if r['peak'] - r['final'] > 0.05)}")
