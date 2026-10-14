"""Draw the README figures from results/*.json and data/windows.npz into figures/."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FS, magnitude

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def load():
    return {r["name"]: r for r in (json.load(open(p)) for p in glob.glob("results/*.json") if "phone_" not in p)}


def fig_models(R):
    names = [("threshold_200hz", "Fixed threshold"), ("rf_200hz", "Random Forest"),
             ("cnn_200hz_mag", "CNN (magnitude)"), ("cnn_200hz_xyz", "CNN (3-axis)")]
    names = [(k, v) for k, v in names if k in R]
    labels = [v for _, v in names]
    sens = [R[k]["sensitivity"] * 100 for k, _ in names]
    fa = [(1 - R[k]["specificity"]) * 100 for k, _ in names]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, vals, title, color in ((axes[0], sens, "Falls detected (%), higher is better", BLUE),
                                   (axes[1], fa, "False alarms (%), lower is better", ORANGE)):
        y = np.arange(len(labels))
        ax.barh(y, vals, height=0.55, color=color)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.1f}", va="center", color=INK, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig("figures/1_model_comparison.png")


def fig_rates(R):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for prefix, label, color in (("rf_", "Random Forest", BLUE), ("cnn_", "CNN (magnitude)", ORANGE)):
        pts = sorted((r["rate_hz"], r["f1"] * 100) for r in R.values()
                     if r["name"].startswith(prefix) and r["input"] in ("features", "mag"))
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_xscale("log", base=2)
    ax.set_xticks([12.5, 25, 50, 100, 200], ["12.5", "25", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("Sampling rate (Hz), lower = less battery")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(90, 100)
    ax.set_title("Cutting the sampling rate 16-fold costs almost no accuracy", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("figures/2_sampling_rate.png")


def fig_age(R):
    names = [("threshold_200hz", "Fixed threshold"), ("rf_200hz", "Random Forest"),
             ("cnn_200hz_mag", "CNN (magnitude)"), ("cnn_200hz_xyz", "CNN (3-axis)")]
    names = [(k, v) for k, v in names if k in R]
    x = np.arange(len(names))
    young = [R[k]["false_alarm_young"] * 100 for k, _ in names]
    old = [R[k]["false_alarm_elderly"] * 100 for k, _ in names]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, young, w, color=BLUE, label="Young adults (19–30)")
    ax.bar(x + w / 2 + 0.01, old, w, color=ORANGE, label="Older adults (60–75)")
    for xi, a, b in zip(x, young, old):
        ax.text(xi - w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=8, color=INK)
        ax.text(xi + w / 2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [v for _, v in names])
    ax.set_ylabel("False alarms (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("False alarms on daily activities: young vs older adults", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("figures/3_young_vs_older.png")


def fig_phone(R):
    """Compare false alarms on SisFall with false alarms on self-recorded phone data."""
    phone = {r["model"]: r for r in (json.load(open(p)) for p in glob.glob("results/phone_*.json"))}
    names = [("threshold_200hz", "Fixed threshold"), ("rf_200hz", "Random Forest"),
             ("cnn_200hz_mag", "CNN (magnitude)"), ("cnn_200hz_xyz", "CNN (3-axis)")]
    names = [(k, v) for k, v in names if k in R and k in phone]
    if not names:
        return False
    x = np.arange(len(names))
    lab = [(1 - R[k]["specificity"]) * 100 for k, _ in names]
    ph = [phone[k]["false_alarms"] / phone[k]["adl"] * 100 for k, _ in names]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, lab, w, color=BLUE, label="SisFall (lab)")
    ax.bar(x + w / 2 + 0.01, ph, w, color=ORANGE, label="Own phone recordings")
    for xi, a, b, (k, _) in zip(x, lab, ph, names):
        ax.text(xi - w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=8, color=INK)
        ax.text(xi + w / 2, b + 1, f"{phone[k]['false_alarms']}/{phone[k]['adl']}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [v for _, v in names])
    ax.set_ylabel("False alarms (%)")
    ax.set_ylim(0, max(ph + lab) * 1.45)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    ax.set_title("False alarms rise sharply from SisFall to a hand-held phone", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("figures/5_phone_test.png")
    return True


def fig_signals():
    d = np.load("data/windows.npz")
    act, subj, X = d["activity"], d["subject"], d["X"]
    t = np.arange(X.shape[1]) / FS
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for code, label, color in (("F01", "Fall (slip, forward)", ORANGE),
                               ("D11", "Sitting down hard on a chair", BLUE)):
        i = np.where((act == code) & (subj == "SA01"))[0]
        if len(i):
            ax.plot(t, magnitude(X[i[0]]), color=color, lw=1.5, label=label)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.text(t[-1], 1.6, "1g (standing still)", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration magnitude (g)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Waist sensor signal: a fall and a hard sit", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("figures/4_example_signals.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    R = load()
    fig_models(R)
    fig_rates(R)
    fig_age(R)
    fig_signals()
    n = 5 if fig_phone(R) else 4
    print(f"Saved {n} figures to figures/")
