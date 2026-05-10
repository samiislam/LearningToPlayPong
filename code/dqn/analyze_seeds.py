"""Aggregate DQN training runs across seeds.

Reads `reward_100` from every `runs/*-pong-dqn-seed*` TensorBoard event file,
reports per-seed sample-efficiency metrics, and saves figures for the paper:

  - learning_curve.png : reward_100 vs total env frames, per seed + mean band,
                        with the threshold line and per-seed frames-to-threshold.
  - auc_bar.png        : per-seed normalised AUC (mean reward over training,
                        evaluated on a common frame budget) + cross-seed mean.

AUC is normalised by the frame budget, so the number is interpretable as the
average reward_100 the agent maintained across training -- directly comparable
between DQN and A2C as long as both are evaluated on the same budget.
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

THRESHOLD = 18.0
TAG = "reward_100"
RUNS_DIR = "runs"


def load_run(path: str) -> tuple[np.ndarray, np.ndarray]:
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    if TAG not in acc.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = acc.Scalars(TAG)
    frames = np.array([e.step for e in events], dtype=np.int64)
    rewards = np.array([e.value for e in events], dtype=np.float64)
    return frames, rewards


def frames_to_threshold(frames: np.ndarray, rewards: np.ndarray,
                        threshold: float) -> int | None:
    hit = np.where(rewards > threshold)[0]
    return int(frames[hit[0]]) if len(hit) else None


def normalised_auc(frames: np.ndarray, rewards: np.ndarray,
                   budget: int) -> float:
    """Mean reward_100 over [0, budget] frames via trapezoidal integration.

    Resampled onto a common grid so per-seed numbers are comparable. Values
    before the first logged episode are filled with the initial reward (i.e.
    we assume the curve starts where it first gets a measurement).
    """
    grid = np.linspace(0, budget, num=1000)
    interp = np.interp(grid, frames, rewards,
                       left=rewards[0], right=rewards[-1])
    return float(np.trapezoid(interp, grid) / budget)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--budget", type=int, default=None,
                        help="Frame budget for AUC. Default: min(max_frame) "
                             "across seeds, so all seeds are compared fairly.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    pattern = os.path.join(args.runs_dir, "*pong-dqn-seed*")
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print(f"No runs matching {pattern}")
        return

    seed_runs: dict[int, list[str]] = defaultdict(list)
    for d in run_dirs:
        m = re.search(r"seed(\d+)", d)
        if m:
            seed_runs[int(m.group(1))].append(d)

    per_seed: dict[int, tuple[np.ndarray, np.ndarray, int | None]] = {}
    for seed, dirs in sorted(seed_runs.items()):
        path = sorted(dirs)[-1]
        frames, rewards = load_run(path)
        if len(frames) == 0:
            print(f"seed {seed}: no {TAG} data in {path}")
            continue
        ftt = frames_to_threshold(frames, rewards, args.threshold)
        per_seed[seed] = (frames, rewards, ftt)

    if not per_seed:
        print("No usable runs.")
        return

    budget = args.budget or min(int(v[0][-1]) for v in per_seed.values())
    print(f"AUC frame budget: {budget:,}\n")

    aucs: dict[int, float] = {}
    print(f"{'seed':>5}  {'episodes':>9}  {'final r100':>11}  "
          f"{'frames>thr':>14}  {'AUC':>8}")
    for seed, (frames, rewards, ftt) in sorted(per_seed.items()):
        auc = normalised_auc(frames, rewards, budget)
        aucs[seed] = auc
        ftt_str = f"{ftt:,}" if ftt is not None else "  not reached"
        print(f"{seed:>5}  {len(rewards):>9}  {rewards[-1]:>11.2f}  "
              f"{ftt_str:>14}  {auc:>8.3f}")

    ftts = [v[2] for v in per_seed.values() if v[2] is not None]
    print()
    if len(ftts) >= 2:
        print(f"Frames-to-threshold (>{args.threshold:.0f}) across "
              f"{len(ftts)} seeds: "
              f"mean={np.mean(ftts):,.0f}  std={np.std(ftts, ddof=1):,.0f}")
    elif len(ftts) == 1:
        print(f"Frames-to-threshold: {ftts[0]:,} (single seed, no std)")
    else:
        print("No seed reached threshold.")
    auc_vals = list(aucs.values())
    if len(auc_vals) >= 2:
        print(f"Normalised AUC across {len(auc_vals)} seeds: "
              f"mean={np.mean(auc_vals):.3f}  "
              f"std={np.std(auc_vals, ddof=1):.3f}")
    else:
        print(f"Normalised AUC: {auc_vals[0]:.3f} (single seed)")

    if args.no_plot:
        return

    import matplotlib.pyplot as plt

    # --- Figure 1: learning curves ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = np.linspace(0, budget, num=1000)
    resampled = []
    for seed, (frames, rewards, ftt) in sorted(per_seed.items()):
        line, = ax.plot(frames, rewards, label=f"seed {seed}", alpha=0.75)
        if ftt is not None:
            ax.axvline(ftt, color=line.get_color(), linestyle=":",
                       linewidth=0.8)
        resampled.append(np.interp(grid, frames, rewards,
                                   left=rewards[0], right=rewards[-1]))
    if len(resampled) >= 2:
        stack = np.stack(resampled)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0, ddof=1)
        ax.plot(grid, mean, color="k", linewidth=1.5, label="mean")
        ax.fill_between(grid, mean - std, mean + std, color="k", alpha=0.15,
                        label=r"$\pm 1\sigma$")
    ax.axhline(args.threshold, color="r", linestyle="--", linewidth=0.8,
               label=f"threshold = {args.threshold:.0f}")
    ax.set_xlabel("Total environment frames")
    ax.set_ylabel("Mean episode reward (last 100)")
    ax.set_title("DQN on Pong: learning curves across seeds")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("learning_curve.png", dpi=150)
    print("\nSaved learning_curve.png")

    # --- Figure 2: AUC bar chart -----------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    seeds_sorted = sorted(aucs.keys())
    vals = [aucs[s] for s in seeds_sorted]
    ax2.bar([str(s) for s in seeds_sorted], vals, color="steelblue",
            edgecolor="k")
    if len(vals) >= 2:
        ax2.axhline(np.mean(vals), color="r", linestyle="--",
                    linewidth=1.0, label=f"mean = {np.mean(vals):.2f}")
        ax2.legend()
    ax2.set_xlabel("Seed")
    ax2.set_ylabel(f"Normalised AUC of reward_100 over {budget:,} frames")
    ax2.set_title("DQN on Pong: per-seed sample-efficiency (AUC)")
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig("auc_bar.png", dpi=150)
    print("Saved auc_bar.png")


if __name__ == "__main__":
    main()
