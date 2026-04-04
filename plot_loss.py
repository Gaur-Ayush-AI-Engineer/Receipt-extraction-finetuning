"""
plot_loss.py — Parse MLX-LM training log and plot loss curves
Usage: python plot_loss.py                      # reads training.log by default
       python plot_loss.py --log training.log   # explicit path
"""

import re
import argparse
import matplotlib.pyplot as plt
import config

# MLX-LM log format:
# Iter 10: Train loss 2.345, Learning Rate 1.000e-04, It/sec 1.23, ...
# Iter 100: Val loss 1.234, Val took 5.678s

TRAIN_RE = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)")
VAL_RE   = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


def parse_log(path: str):
    train_iters, train_losses = [], []
    val_iters, val_losses = [], []

    with open(path) as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                train_iters.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue
            m = VAL_RE.search(line)
            if m:
                val_iters.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))

    return train_iters, train_losses, val_iters, val_losses


def plot(train_iters, train_losses, val_iters, val_losses, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(train_iters, train_losses, label="Train loss", linewidth=1.5, alpha=0.85)
    if val_iters:
        ax.plot(val_iters, val_losses, label="Val loss", linewidth=2, marker="o", markersize=4)

    ax.set_title("Qwen2.5-3B LoRA Training — Receipt Extraction", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="training.log", help="Path to training log file")
    parser.add_argument("--out", default=f"{config.RESULTS_DIR}/training_loss.png")
    args = parser.parse_args()

    train_iters, train_losses, val_iters, val_losses = parse_log(args.log)

    if not train_losses:
        print("No training loss entries found in log. Check the log format.")
        return

    print(f"Parsed {len(train_losses)} train points, {len(val_losses)} val points.")
    plot(train_iters, train_losses, val_iters, val_losses, args.out)


if __name__ == "__main__":
    main()
