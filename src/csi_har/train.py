"""Cross-subject training and evaluation.

Usage:
    python -m csi_har.train --data data/processed.npz

1. 5-fold GroupKFold by subject: each fold tests on subjects never seen in training.
   Early stopping uses separate validation subjects, never the test fold.
2. Out-of-fold predictions give the reported metrics (overall, per fold, per environment).
3. A final model is trained on all subjects (minus validation subjects) and saved for the API.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from csi_har.config import CLASS_NAMES, PreprocessConfig, TrainConfig
from csi_har.data.dataset import load_cache
from csi_har.evaluate import compute_metrics, plot_confusion
from csi_har.inference import save_model
from csi_har.models import HARCNN
from csi_har.splits import cross_subject_folds


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Light augmentation: random gain per sample and Gaussian jitter."""
    gain = 1 + 0.1 * torch.randn(x.shape[0], 1, 1)
    return x * gain + 0.05 * torch.randn_like(x)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i : i + batch_size])).argmax(1) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


def fit(x_tr, y_tr, x_val, y_val, cfg: TrainConfig, log_prefix: str = "") -> tuple[HARCNN, int]:
    model = HARCNN()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(augment(xb)), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        val_acc = float((predict(model, x_val) == y_val).mean())
        print(f"{log_prefix}epoch {epoch:2d} loss {total / len(x_tr):.3f} val_acc {val_acc:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def run(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    data = load_cache(cfg.data_path)
    x, y = data["X"], (data["activity"] - 1).astype(np.int64)
    subjects, envs = data["subject"], data["environment"]
    print(f"{len(x)} trials, {len(np.unique(subjects))} subjects, input {x.shape[1:]}")

    oof = np.full(len(y), -1)
    folds = []
    for k, (tr, val, te) in enumerate(cross_subject_folds(subjects, cfg.n_folds, cfg.val_subjects, cfg.seed), 1):
        model, best_epoch = fit(x[tr], y[tr], x[val], y[val], cfg, log_prefix=f"[fold {k}] ")
        oof[te] = predict(model, x[te])
        m = compute_metrics(y[te], oof[te], CLASS_NAMES)
        folds.append({"fold": k, "test_subjects": sorted(map(int, np.unique(subjects[te]))),
                      "best_epoch": best_epoch, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        print(f"[fold {k}] test acc {m['accuracy']:.3f} macro-F1 {m['macro_f1']:.3f}", flush=True)

    overall = compute_metrics(y, oof, CLASS_NAMES)
    accs, f1s = [f["accuracy"] for f in folds], [f["macro_f1"] for f in folds]
    by_env = {f"environment_{e}": float((oof[envs == e] == y[envs == e]).mean()) for e in np.unique(envs)}
    metrics = {
        "protocol": f"{cfg.n_folds}-fold cross-subject (GroupKFold by subject), out-of-fold predictions",
        "n_trials": int(len(y)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "accuracy_by_environment": by_env,
        "folds": folds,
        "overall": overall,
    }
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(np.array(overall["confusion_matrix"]), CLASS_NAMES, cfg.report_dir / "confusion_matrix.png",
                   f"Cross-subject confusion matrix (accuracy {overall['accuracy']:.1%})")

    # Final model for serving: all subjects except a few held out for early stopping.
    rng = np.random.default_rng(cfg.seed)
    val_ids = rng.choice(np.unique(subjects), size=cfg.val_subjects, replace=False)
    is_val = np.isin(subjects, val_ids)
    final, best_epoch = fit(x[~is_val], y[~is_val], x[is_val], y[is_val], cfg, log_prefix="[final] ")
    save_model(final, cfg.model_dir / "har_cnn.pt", PreprocessConfig(target_length=x.shape[2]),
               extra={"cv_accuracy_mean": metrics["accuracy_mean"], "cv_macro_f1_mean": metrics["macro_f1_mean"],
                      "final_best_epoch": best_epoch})
    print(json.dumps({k: metrics[k] for k in ("accuracy_mean", "accuracy_std", "macro_f1_mean",
                                              "macro_f1_std", "accuracy_by_environment")}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = TrainConfig()
    parser.add_argument("--data", type=Path, default=defaults.data_path)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--folds", type=int, default=defaults.n_folds)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    run(TrainConfig(data_path=args.data, epochs=args.epochs, n_folds=args.folds, batch_size=args.batch_size,
                    lr=args.lr, seed=args.seed, num_threads=args.threads))


if __name__ == "__main__":
    main()
