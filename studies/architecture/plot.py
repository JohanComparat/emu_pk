"""Figure of the architecture scan: test error against size and against data.

Reads the ``<tag>.json`` files written by ``scan.py`` and draws
(a) the median test error at z = 0 against the number of weights, on the full
training pool, one line per depth, and (b) the same error against the number of
training cosmologies, one line per architecture.  The dashed line is the median
distance of the training truth from CAMB's converged answer (0.039 %,
docs/reproducing.md), below which a smaller test error does not improve the
emulator.

Run: python studies/architecture/plot.py --scan ~/data/emu_pk_architecture_scan --out arch_scan.pdf
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

TRUTH_FLOOR = 0.039          # per cent, training truth against converged CAMB
FULL = 145475
DEPTH_COLOURS = {1: "#86b6ef", 2: "#3987e5", 4: "#1c5cab", 6: "#0d366b"}
DEPTH_MARKERS = {1: "o", 2: "s", 4: "D", 6: "^"}
ARCH_COLOURS = {"1024": "#2a78d6", "512x512": "#eb6834", "512x512x512x512": "#1baf7a",
                "1024x1024x1024x1024": "#eda100", "256x256x256x256": "#e87ba4",
                "512x512x512x512x512x512": "#008300"}


def label(h):
    return f"{len(h)}$\\times${h[0]}" if len(h) > 1 else f"[{h[0]}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="~/data/emu_pk_architecture_scan")
    ap.add_argument("--out", default="arch_scan.pdf")
    a = ap.parse_args()
    runs = [json.loads(p.read_text())
            for p in pathlib.Path(a.scan).expanduser().glob("h*.json")]
    err = lambda r: 100 * r["test_z0"]["median"]

    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                         "savefig.bbox": "tight"})
    scaling = any(sum("x".join(map(str, r["hidden"])) == key for r in runs) > 1
                  for key in ARCH_COLOURS)
    if scaling:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    else:                    # the data-scaling runs are not in yet
        fig, ax1 = plt.subplots(figsize=(5.5, 3.8))

    full = [r for r in runs if r["n_cos"] == FULL]
    for d, c in DEPTH_COLOURS.items():
        rr = sorted((r for r in full if r["depth"] == d), key=lambda r: r["n_par"])
        if not rr:
            continue
        ax1.plot([r["n_par"] for r in rr], [err(r) for r in rr], "-", marker=DEPTH_MARKERS[d],
                 color=c, lw=2,
                 ms=6, label=f"{d} hidden layer{'s' if d > 1 else ''}")
        for r in rr:
            ax1.annotate(str(r["hidden"][0]), (r["n_par"], err(r)), fontsize=7,
                         xytext=(3, 3), textcoords="offset points", color="0.3")
    ax1.axhline(TRUTH_FLOOR, ls="--", color="0.4", lw=1)
    ax1.text(0.02, TRUTH_FLOOR * 1.08, "training truth vs converged CAMB", fontsize=8,
             color="0.3", transform=ax1.get_yaxis_transform())
    ax1.set(xscale="log", yscale="log", xlabel="number of weights",
            ylabel=r"median max$_k|\Delta P/P|$ at $z=0$ [%]",
            title=f"{FULL:,} training cosmologies".replace(",", "\u2009"))
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.legend(fontsize=8, loc="upper right")

    for key, c in ARCH_COLOURS.items():
        if not scaling:
            break
        rr = sorted((r for r in runs if "x".join(map(str, r["hidden"])) == key),
                    key=lambda r: r["n_cos"])
        if len(rr) < 2:
            continue
        ax2.plot([r["n_cos"] for r in rr], [err(r) for r in rr], "o-", color=c, lw=2,
                 ms=6, label=f"{label(rr[0]['hidden'])} ({rr[0]['n_par'] / 1e6:.2f}M)")
    if scaling:
        ax2.axhline(TRUTH_FLOOR, ls="--", color="0.4", lw=1)
        ax2.set(xscale="log", yscale="log", xlabel="training cosmologies (31 redshifts each)",
                ylabel=r"median max$_k|\Delta P/P|$ at $z=0$ [%]", title="data scaling")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"{len(runs)} runs -> {a.out}")


if __name__ == "__main__":
    main()
