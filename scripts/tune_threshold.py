"""
Threshold Tuning Script for Drone Audio Detection
Finds optimal classification threshold based on F1-max on validation set
Usage:
    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --model models/drone_model_xxx.keras
    python scripts/tune_threshold.py --plot --no-show
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import load_processed_data, split_by_groups


def find_latest_model(models_dir):
    models = sorted(Path(models_dir).glob("drone_model_*.keras"))
    if not models:
        models = sorted(Path(models_dir).glob("*.keras"))
    if not models:
        print(f"No .keras files found in {models_dir}")
        sys.exit(1)
    latest = models[-1]
    print(f"Model selected: {latest.name}")
    return str(latest)


def tune_threshold(model_path, data_dir, output_plot=None, show_plot=True):
    print("=" * 60)
    print("THRESHOLD TUNING - DRONE DETECTION")
    print("=" * 60)

    print("\n1. Loading model...")
    model = tf.keras.models.load_model(model_path)
    print(f"   Model loaded: {os.path.basename(model_path)}")

    print("\n2. Loading processed data...")
    X, y, groups = load_processed_data(data_dir)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    print(f"   Reshaped features: {X.shape}")

    print("\n3. Splitting dataset (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_by_groups(
        X, y, groups, train_size=0.7, val_size=0.15, test_size=0.15
    )
    print(f"   Validation set: {len(X_val)} samples")
    drone_count = int(np.sum(y_val == 1))
    bg_count = int(np.sum(y_val == 0))
    print(f"   DRONE: {drone_count}, NOT_DRONE: {bg_count}")

    print("\n4. Running inference on validation set...")
    y_pred_proba = model.predict(X_val, verbose=0).flatten()

    thresholds = np.arange(0.05, 0.96, 0.05)
    results = []

    print("\n5. Evaluating thresholds...")
    for t in thresholds:
        y_pred = (y_pred_proba > t).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        results.append((t, prec, rec, f1))

    results = np.array(results)
    best_idx = np.argmax(results[:, 3])
    best_threshold, best_prec, best_rec, best_f1 = results[best_idx]

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 60)
    print(f" {'Threshold':>9} | {'Precision':>9} | {'Recall':>9} | {'F1':>9}")
    print("-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+-" + "-" * 9)
    for i, (t, prec, rec, f1) in enumerate(results):
        marker = " *" if i == best_idx else "  "
        print(f" {t:>7.2f}{marker} | {prec:>9.4f} | {rec:>9.4f} | {f1:>9.4f}")

    print("\n" + "=" * 60)
    print(f"  BEST THRESHOLD (F1-max): {best_threshold:.2f}")
    print(f"  Precision: {best_prec:.4f}")
    print(f"  Recall:    {best_rec:.4f}")
    print(f"  F1-score:  {best_f1:.4f}")
    print(f"  Current threshold (code): 0.50")
    delta = best_threshold - 0.50
    if abs(delta) > 0.01:
        direction = "increase" if delta > 0 else "decrease"
        print(f"  >> Recommend to {direction} threshold by {abs(delta):.2f}")
    print("=" * 60)

    # Confusion matrix at best threshold
    y_pred_best = (y_pred_proba > best_threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred_best, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix @ threshold = {best_threshold:.2f}:")
    print(f"  [[{tn:>4}  {fp:>4}]  (TN, FP)")
    print(f"   [{fn:>4}  {tp:>4}]]  (FN, TP)")
    print(f"\n  Accuracy:  {(tn + tp) / (tn + fp + fn + tp):.4f}")
    print(f"  Precision: {tp / (tp + fp):.4f}" if (tp + fp) > 0 else "  Precision: N/A")
    print(f"  Recall:    {tp / (tp + fn):.4f}" if (tp + fn) > 0 else "  Recall:    N/A")

    # Plot
    if output_plot or show_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: PR curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_pred_proba)
        axes[0].plot(rec_curve, prec_curve, 'b-', linewidth=2)
        axes[0].scatter([best_rec], [best_prec], color='red', s=80, zorder=5,
                        label=f'Best threshold = {best_threshold:.2f}')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1.05)
        axes[0].set_ylim(0, 1.05)

        # Right: Metrics vs Threshold
        ts = results[:, 0]
        axes[1].plot(ts, results[:, 1], 'g-', label='Precision', linewidth=2)
        axes[1].plot(ts, results[:, 2], 'r-', label='Recall', linewidth=2)
        axes[1].plot(ts, results[:, 3], 'b-', label='F1-score', linewidth=2)
        axes[1].axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.7)
        axes[1].axvline(x=0.50, color='orange', linestyle=':', alpha=0.5, label='Current (0.50)')
        axes[1].scatter([best_threshold], [best_f1], color='red', s=80, zorder=5)
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Metrics vs. Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1)

        plt.tight_layout()

        if output_plot:
            os.makedirs(os.path.dirname(output_plot) or '.', exist_ok=True)
            plt.savefig(output_plot, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {output_plot}")

        if show_plot:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal threshold for drone detection model"
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to .keras model file (default: latest in models/)'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to processed data directory (default: data/processed)'
    )
    parser.add_argument(
        '--plot', type=str, default=None, nargs='?', const='auto',
        help='Save plot to file (default: models/threshold_tuning_<timestamp>.png)'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display the plot interactively'
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / 'models'
    data_dir = Path(args.data) if args.data else (project_root / 'data' / 'processed')

    model_path = args.model if args.model else find_latest_model(models_dir)

    output_plot = None
    if args.plot == 'auto':
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_plot = str(models_dir / f"threshold_tuning_{ts}.png")
    elif args.plot:
        output_plot = args.plot

    tune_threshold(model_path, str(data_dir),
                   output_plot=output_plot,
                   show_plot=not args.no_show)


if __name__ == "__main__":
    main()
