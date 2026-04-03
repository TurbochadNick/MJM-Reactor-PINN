#!/usr/bin/env python
from __future__ import annotations

import json
import math
import sys
import time
import types
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path("/Users/jerrychristensen/Downloads")
NOTEBOOK_PATH = DOWNLOADS / "612_code_v1.ipynb"
SEED_DATA_PATH = DOWNLOADS / "pinn_training_data.json"
ARTIFACT_DIR = ROOT / "artifacts" / "first_pinn_run"

INPUT_KEYS = [
    "enrichment",
    "uf4_mol_frac",
    "radius",
    "height",
    "temperature",
    "water_vol_frac",
]
TARGET_KEYS = [
    "k_eff",
    "k_inf",
    "peaking_factor",
    "alpha_pcm_per_K",
]


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_v1_solver() -> dict:
    notebook = json.loads(NOTEBOOK_PATH.read_text())
    source = "".join(notebook["cells"][0]["source"])
    module_name = "mjm_solver_v1"
    module = types.ModuleType(module_name)
    module.__file__ = str(NOTEBOOK_PATH)
    sys.modules[module_name] = module
    namespace = module.__dict__
    namespace["__name__"] = module_name
    exec(source, namespace)
    return namespace


def load_seed_data() -> list[dict]:
    return json.loads(SEED_DATA_PATH.read_text())


def geometric_buckling(radius_cm: float, height_cm: float) -> tuple[float, float, float]:
    br2 = (2.405 / max(radius_cm, 1.0e-6)) ** 2
    bz2 = (math.pi / max(height_cm, 1.0e-6)) ** 2
    return br2, bz2, br2 + bz2


def augment_with_alpha(
    solver_ns: dict,
    rows: list[dict],
    force_recompute: bool = False,
) -> list[dict]:
    compute_temperature_coefficient = solver_ns["compute_temperature_coefficient"]
    augmented: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        record = dict(row)
        if force_recompute or "alpha_pcm_per_K" not in record:
            params = {key: float(record[key]) for key in INPUT_KEYS}
            alpha = compute_temperature_coefficient(params)
            record["alpha_pcm_per_K"] = float(alpha["alpha_pcm_per_K"])
        br2, bz2, bg2 = geometric_buckling(float(record["radius"]), float(record["height"]))
        record["br2"] = br2
        record["bz2"] = bz2
        record["bg2"] = bg2
        augmented.append(to_builtin(record))

        if idx % 50 == 0 or idx == len(rows):
            print(f"  augmented {idx}/{len(rows)} samples")

    return augmented


def build_dataset(solver_ns: dict, n_total_samples: int, seed: int) -> list[dict]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    augmented_path = ARTIFACT_DIR / f"augmented_dataset_{n_total_samples}.json"
    if augmented_path.exists():
        print(f"  loading cached augmented dataset: {augmented_path.name}")
        return json.loads(augmented_path.read_text())

    seed_rows = load_seed_data()

    if n_total_samples <= len(seed_rows):
        rows = seed_rows[:n_total_samples]
    else:
        n_extra = n_total_samples - len(seed_rows)
        print(f"  generating {n_extra} additional solver samples")
        np.random.seed(seed)
        generated_rows = solver_ns["generate_training_data"](
            n_samples=n_extra,
            output_path=None,
            verbose=False,
        )
        rows = seed_rows + [to_builtin(row) for row in generated_rows]

    print(f"  computing alpha_T and buckling features for {len(rows)} samples")
    augmented = augment_with_alpha(solver_ns, rows)
    augmented_path.write_text(json.dumps(augmented, indent=2))
    return augmented


def dataset_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = np.array([[float(row[key]) for key in INPUT_KEYS] for row in rows], dtype=float)
    x_buckling = np.array(
        [[float(row["br2"]), float(row["bz2"]), float(row["bg2"])] for row in rows],
        dtype=float,
    )
    y = np.array([[float(row[key]) for key in TARGET_KEYS] for row in rows], dtype=float)
    return x_raw, x_buckling, y


def train_test_split(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_inv(x: float) -> float:
    if x < 1.0e-6:
        return math.log(1.0e-6)
    return math.log(math.expm1(x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    data_loss: list[float]
    physics_loss: list[float]
    alpha_penalty: list[float]


class ScalarPINN:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, int] = (64, 64),
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        h1, h2 = hidden_dims
        self.params = {
            "W1": self.rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, h1)),
            "b1": np.zeros((1, h1)),
            "W2": self.rng.normal(scale=np.sqrt(2.0 / h1), size=(h1, h2)),
            "b2": np.zeros((1, h2)),
            "W3": self.rng.normal(scale=np.sqrt(2.0 / h2), size=(h2, output_dim)),
            "b3": np.zeros((1, output_dim)),
            "log_m2": np.array([[softplus_inv(25.0)]], dtype=float),
        }
        self.mom1 = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.mom2 = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.best_params: dict[str, np.ndarray] | None = None
        self.best_val_loss = float("inf")

    def _prepare(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_mean is None:
            self.x_mean = x.mean(axis=0, keepdims=True)
            self.x_std = x.std(axis=0, keepdims=True) + 1.0e-8
            self.y_mean = y.mean(axis=0, keepdims=True)
            self.y_std = y.std(axis=0, keepdims=True) + 1.0e-8
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        return x_norm, y_norm

    def _forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x_norm @ self.params["W1"] + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.tanh(z2)
        y_norm = a2 @ self.params["W3"] + self.params["b3"]
        cache = {"x_norm": x_norm, "a1": a1, "a2": a2}
        return y_norm, cache

    def _loss_and_grad(
        self,
        y_pred_norm: np.ndarray,
        y_true_norm: np.ndarray,
        bg2: np.ndarray,
        lambda_physics: float,
        lambda_alpha: float,
        lambda_l2: float,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        assert self.y_mean is not None
        assert self.y_std is not None

        diff = y_pred_norm - y_true_norm
        data_loss = float(np.mean(diff ** 2))
        grad_y_norm = (2.0 / diff.size) * diff

        y_pred_phys = y_pred_norm * self.y_std + self.y_mean
        k_eff = y_pred_phys[:, 0]
        k_inf = y_pred_phys[:, 1]
        alpha = y_pred_phys[:, 3]

        m2 = softplus(self.params["log_m2"])[0, 0] + 1.0e-8
        denom = 1.0 + m2 * bg2
        criticality_proxy = k_inf / denom
        physics_residual = k_eff - criticality_proxy
        physics_loss = float(np.mean(physics_residual ** 2))

        d_physics_phys = np.zeros_like(y_pred_phys)
        d_physics_phys[:, 0] = (2.0 / len(bg2)) * physics_residual
        d_physics_phys[:, 1] = -(2.0 / len(bg2)) * physics_residual / denom
        grad_y_norm += lambda_physics * d_physics_phys * self.y_std

        dm2 = (2.0 / len(bg2)) * np.sum(
            physics_residual * (k_inf * bg2 / (denom ** 2))
        )
        dlog_m2 = lambda_physics * dm2 * sigmoid(self.params["log_m2"])

        alpha_positive = np.maximum(alpha, 0.0)
        alpha_penalty = float(np.mean(alpha_positive ** 2))
        d_alpha_phys = (2.0 / len(alpha_positive)) * alpha_positive
        grad_y_norm[:, 3] += lambda_alpha * d_alpha_phys * self.y_std[0, 3]

        total_loss = data_loss + lambda_physics * physics_loss + lambda_alpha * alpha_penalty

        reg_loss = 0.0
        for name in ("W1", "W2", "W3"):
            reg_loss += float(np.sum(self.params[name] ** 2))
        total_loss += lambda_l2 * reg_loss

        losses = {
            "total": total_loss,
            "data": data_loss,
            "physics": physics_loss,
            "alpha": alpha_penalty,
            "m2": float(m2),
        }
        return losses, grad_y_norm, dlog_m2

    def _backprop(
        self,
        cache: dict[str, np.ndarray],
        grad_y_norm: np.ndarray,
        dlog_m2: np.ndarray,
        lambda_l2: float,
    ) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        a2 = cache["a2"]
        a1 = cache["a1"]
        x_norm = cache["x_norm"]

        grads["W3"] = a2.T @ grad_y_norm + 2.0 * lambda_l2 * self.params["W3"]
        grads["b3"] = np.sum(grad_y_norm, axis=0, keepdims=True)

        da2 = grad_y_norm @ self.params["W3"].T
        dz2 = da2 * (1.0 - a2 ** 2)
        grads["W2"] = a1.T @ dz2 + 2.0 * lambda_l2 * self.params["W2"]
        grads["b2"] = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (1.0 - a1 ** 2)
        grads["W1"] = x_norm.T @ dz1 + 2.0 * lambda_l2 * self.params["W1"]
        grads["b1"] = np.sum(dz1, axis=0, keepdims=True)

        grads["log_m2"] = dlog_m2
        return grads

    def _apply_adam(self, grads: dict[str, np.ndarray], lr: float, step: int) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1.0e-8
        for name, grad in grads.items():
            self.mom1[name] = beta1 * self.mom1[name] + (1.0 - beta1) * grad
            self.mom2[name] = beta2 * self.mom2[name] + (1.0 - beta2) * (grad ** 2)
            m_hat = self.mom1[name] / (1.0 - beta1 ** step)
            v_hat = self.mom2[name] / (1.0 - beta2 ** step)
            self.params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        bg2_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        bg2_val: np.ndarray,
        *,
        epochs: int,
        lr: float,
        lambda_physics: float,
        lambda_alpha: float,
        lambda_l2: float,
        print_every: int = 100,
    ) -> TrainingHistory:
        x_train_norm, y_train_norm = self._prepare(x_train, y_train)
        x_val_norm = (x_val - self.x_mean) / self.x_std
        y_val_norm = (y_val - self.y_mean) / self.y_std

        history = TrainingHistory([], [], [], [], [])
        for epoch in range(1, epochs + 1):
            y_pred_norm, cache = self._forward(x_train_norm)
            train_losses, grad_y_norm, dlog_m2 = self._loss_and_grad(
                y_pred_norm,
                y_train_norm,
                bg2_train,
                lambda_physics=lambda_physics,
                lambda_alpha=lambda_alpha,
                lambda_l2=lambda_l2,
            )
            grads = self._backprop(cache, grad_y_norm, dlog_m2, lambda_l2=lambda_l2)
            self._apply_adam(grads, lr=lr, step=epoch)

            val_pred_norm, _ = self._forward(x_val_norm)
            val_losses, _, _ = self._loss_and_grad(
                val_pred_norm,
                y_val_norm,
                bg2_val,
                lambda_physics=lambda_physics,
                lambda_alpha=lambda_alpha,
                lambda_l2=lambda_l2,
            )
            history.train_loss.append(train_losses["total"])
            history.val_loss.append(val_losses["total"])
            history.data_loss.append(train_losses["data"])
            history.physics_loss.append(train_losses["physics"])
            history.alpha_penalty.append(train_losses["alpha"])

            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.best_params = deepcopy(self.params)

            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(
                    f"  epoch {epoch:4d} | train {train_losses['total']:.5f} "
                    f"| val {val_losses['total']:.5f} | m2 {val_losses['m2']:.3f}"
                )

        if self.best_params is not None:
            self.params = deepcopy(self.best_params)
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.x_mean is not None
        assert self.x_std is not None
        assert self.y_mean is not None
        assert self.y_std is not None
        x_norm = (x - self.x_mean) / self.x_std
        y_norm, _ = self._forward(x_norm)
        return y_norm * self.y_std + self.y_mean

    def learned_m2(self) -> float:
        return float(softplus(self.params["log_m2"])[0, 0])


def metrics_for_targets(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = {}
    for idx, key in enumerate(TARGET_KEYS):
        err = y_pred[:, idx] - y_true[:, idx]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.sum((y_true[:, idx] - y_true[:, idx].mean()) ** 2)
        r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
        result[key] = {"rmse": rmse, "mae": mae, "r2": r2}
    return result


def physics_residual(y_pred: np.ndarray, bg2: np.ndarray, m2: float) -> float:
    resid = y_pred[:, 0] - y_pred[:, 1] / (1.0 + m2 * bg2)
    return float(np.sqrt(np.mean(resid ** 2)))


def summarize_upgrades(
    baseline_metrics: dict,
    pinn_metrics: dict,
    baseline_phys_rmse: float,
    pinn_phys_rmse: float,
    n_samples: int,
) -> list[str]:
    return [
        f"- Dataset expanded to {n_samples} labeled samples with alpha_T added from the v1 solver.",
        f"- k_eff test RMSE moved from {baseline_metrics['k_eff']['rmse']:.5f} to {pinn_metrics['k_eff']['rmse']:.5f}.",
        f"- k_inf test RMSE moved from {baseline_metrics['k_inf']['rmse']:.5f} to {pinn_metrics['k_inf']['rmse']:.5f}.",
        (
            "- alpha_T test RMSE moved from "
            f"{baseline_metrics['alpha_pcm_per_K']['rmse']:.2f} pcm/K to "
            f"{pinn_metrics['alpha_pcm_per_K']['rmse']:.2f} pcm/K."
        ),
        (
            "- criticality-consistency RMSE "
            f"k_eff - k_inf/(1 + M^2 Bg^2) moved from {baseline_phys_rmse:.5f} "
            f"to {pinn_phys_rmse:.5f}."
        ),
        (
            "- The v2 model also learns alpha_T directly and uses finite-cylinder "
            "buckling features Br^2, Bz^2, and Bg^2 in the input space."
        ),
    ]


def save_parity_plot(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    pinn_pred: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    pairs = [("k_eff", 0), ("alpha_pcm_per_K", 3)]
    for ax, (label, idx) in zip(axes, pairs, strict=True):
        truth = y_true[:, idx]
        ax.scatter(truth, baseline_pred[:, idx], s=20, alpha=0.7, label="baseline")
        ax.scatter(truth, pinn_pred[:, idx], s=20, alpha=0.7, label="pinn-lite")
        lo = min(truth.min(), baseline_pred[:, idx].min(), pinn_pred[:, idx].min())
        hi = max(truth.max(), baseline_pred[:, idx].max(), pinn_pred[:, idx].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_loss_plot(
    baseline_history: TrainingHistory,
    pinn_history: TrainingHistory,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(baseline_history.train_loss, label="baseline train")
    axes[0].plot(baseline_history.val_loss, label="baseline val")
    axes[0].set_title("Baseline Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(pinn_history.train_loss, label="pinn train")
    axes[1].plot(pinn_history.val_loss, label="pinn val")
    axes[1].plot(pinn_history.physics_loss, label="pinn physics")
    axes[1].set_title("PINN-lite Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(summary_path: Path, content: dict) -> None:
    lines = [
        "# First PINN Run",
        "",
        "## Run Setup",
        "",
        f"- Total samples: {content['n_samples']}",
        "- Targets: k_eff, k_inf, peaking_factor, alpha_T",
        "- Baseline model: plain neural surrogate on 6 raw design inputs",
        "- Upgraded model: buckling-aware physics-guided PINN-lite on 9 inputs",
        "",
        "## Upgrades",
        "",
    ]
    lines.extend(content["upgrade_lines"])
    lines.extend(
        [
            "",
            "## Learned Physics Parameter",
            "",
            f"- PINN-lite learned M^2 leakage parameter: {content['pinn_learned_m2']:.4f}",
            "",
            "## Artifact Files",
            "",
            "- parity_plot.png",
            "- loss_plot.png",
            "- metrics.json",
            "- augmented_dataset_*.json",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    t0 = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading v1 solver from notebook...")
    solver_ns = load_v1_solver()

    n_total_samples = 400
    seed = 42
    rows = build_dataset(solver_ns, n_total_samples=n_total_samples, seed=seed)
    x_raw, x_buckling, y = dataset_to_arrays(rows)
    x_pinn = np.concatenate([x_raw, x_buckling], axis=1)
    bg2 = x_buckling[:, 2]

    train_idx, test_idx = train_test_split(len(rows), test_fraction=0.2, seed=seed)

    x_train_raw = x_raw[train_idx]
    x_test_raw = x_raw[test_idx]
    x_train_pinn = x_pinn[train_idx]
    x_test_pinn = x_pinn[test_idx]
    bg2_train = bg2[train_idx]
    bg2_test = bg2[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("\nTraining baseline neural surrogate...")
    baseline = ScalarPINN(input_dim=x_train_raw.shape[1], output_dim=y_train.shape[1], seed=seed)
    baseline_history = baseline.fit(
        x_train_raw,
        y_train,
        bg2_train,
        x_test_raw,
        y_test,
        bg2_test,
        epochs=1200,
        lr=3.0e-3,
        lambda_physics=0.0,
        lambda_alpha=0.0,
        lambda_l2=1.0e-5,
        print_every=200,
    )

    print("\nTraining buckling-aware PINN-lite...")
    pinn = ScalarPINN(input_dim=x_train_pinn.shape[1], output_dim=y_train.shape[1], seed=seed)
    pinn_history = pinn.fit(
        x_train_pinn,
        y_train,
        bg2_train,
        x_test_pinn,
        y_test,
        bg2_test,
        epochs=1500,
        lr=2.5e-3,
        lambda_physics=0.35,
        lambda_alpha=0.01,
        lambda_l2=1.0e-5,
        print_every=200,
    )

    baseline_pred = baseline.predict(x_test_raw)
    pinn_pred = pinn.predict(x_test_pinn)

    baseline_metrics = metrics_for_targets(y_test, baseline_pred)
    pinn_metrics = metrics_for_targets(y_test, pinn_pred)
    baseline_phys_rmse = physics_residual(baseline_pred, bg2_test, baseline.learned_m2())
    pinn_phys_rmse = physics_residual(pinn_pred, bg2_test, pinn.learned_m2())

    parity_path = ARTIFACT_DIR / "parity_plot.png"
    loss_path = ARTIFACT_DIR / "loss_plot.png"
    metrics_path = ARTIFACT_DIR / "metrics.json"
    summary_path = ARTIFACT_DIR / "summary.md"

    save_parity_plot(y_test, baseline_pred, pinn_pred, parity_path)
    save_loss_plot(baseline_history, pinn_history, loss_path)

    upgrade_lines = summarize_upgrades(
        baseline_metrics=baseline_metrics,
        pinn_metrics=pinn_metrics,
        baseline_phys_rmse=baseline_phys_rmse,
        pinn_phys_rmse=pinn_phys_rmse,
        n_samples=len(rows),
    )
    metrics_payload = {
        "n_samples": len(rows),
        "targets": TARGET_KEYS,
        "baseline": {
            "metrics": baseline_metrics,
            "physics_residual_rmse": baseline_phys_rmse,
            "learned_m2": baseline.learned_m2(),
        },
        "pinn_lite": {
            "metrics": pinn_metrics,
            "physics_residual_rmse": pinn_phys_rmse,
            "learned_m2": pinn.learned_m2(),
        },
        "upgrade_lines": upgrade_lines,
        "runtime_seconds": time.time() - t0,
    }
    metrics_path.write_text(json.dumps(to_builtin(metrics_payload), indent=2))
    write_summary(
        summary_path,
        {
            "n_samples": len(rows),
            "upgrade_lines": upgrade_lines,
            "pinn_learned_m2": pinn.learned_m2(),
        },
    )

    print("\nRun complete.")
    print(f"  summary: {summary_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  parity plot: {parity_path}")
    print(f"  loss plot: {loss_path}")


if __name__ == "__main__":
    main()
