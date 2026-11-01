import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt

# plots train/val loss from the CSVs that train.py writes to logs/
# usage:
#   python plot_losses.py                    # latest run
#   python plot_losses.py logs/a.csv logs/b.csv   # compare runs
#   python plot_losses.py --all              # every run in logs/
#   python plot_losses.py --ymax 3           # zoom in past the first steps

def read_log(path):
    cols = {"tokens": [], "train_loss": [], "val_loss": [], "lr": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            for key in cols:
                cols[key].append(float(row[key]))
    return cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="*", help="CSV files to plot (default: latest in logs/)")
    parser.add_argument("--all", action="store_true", help="plot every run in logs/")
    parser.add_argument("--out", default="loss_curves.png", help="where to save the figure")
    # the first evals (~7.7) squash the rest of the curve, so the y-axis can be capped
    parser.add_argument("--ymax", type=float, default=None, help="top of the loss axis")
    args = parser.parse_args()

    paths = args.logs or sorted(glob.glob("logs/*.csv"))
    if not args.logs and not args.all:
        paths = paths[-1:]
    if not paths:
        raise SystemExit("No logs found. Run train.py first.")

    fig, (ax_loss, ax_lr) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for path in paths:
        log = read_log(path)
        name = os.path.splitext(os.path.basename(path))[0]
        # tokens on the x-axis, so runs with different batch or context sizes line up
        tokens_m = [t / 1e6 for t in log["tokens"]]
        line, = ax_loss.plot(tokens_m, log["val_loss"], label=f"{name} val")
        ax_loss.plot(tokens_m, log["train_loss"], linestyle="--", alpha=0.6,
                     color=line.get_color(), label=f"{name} train")
        ax_lr.plot(tokens_m, log["lr"], color=line.get_color())
        best = min(range(len(log["val_loss"])), key=log["val_loss"].__getitem__)
        print(f"{name}: best val {log['val_loss'][best]:.3f} at "
              f"{log['tokens'][best] / 1e6:.1f}M tokens")

    if args.ymax is not None:
        ax_loss.set_ylim(top=args.ymax)
    ax_loss.set_ylabel("loss (nats/token)")
    ax_loss.set_title("Training and validation loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("tokens seen (millions)")
    ax_lr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")
