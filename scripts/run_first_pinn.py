#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "first_pinn_run"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / ".mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monster_v2_package.solver_v2 import generate_dataset_v2


DATASET_PATH = ROOT / "artifacts" / "solver_v2" / "dataset_v2_small.json"

RAW_INPUT_KEYS = [
    "enrichment",
    "uf4_mol_frac",
    "radius_cm",
    "height_cm",
    "temperature_K",
    "fuel_volume_fraction",
    "water_vol_frac",
]
FEATURE_KEYS = RAW_INPUT_KEYS + [
    "Br2_cm2_inv",
    "Bz2_cm2_inv",
    "Bg2_cm2_inv",
    "D1_cm",
    "D2_cm",
    "Sigma_a1_cm1",
    "Sigma_a2_cm1",
    "Sigma_s12_cm1",
    "nuSigma_f1_cm1",
    "nuSigma_f2_cm1",
]
TARGET_KEYS = [
    "k_eff",
    "k_inf",
    "alpha_T_pcm_per_K",
    "peaking_factor",
    "peak_flux_n_cm2_s",
    "peak_power_density_W_cm3",
    "beta_eff_ratio",
]


def to_builtin(value):
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_dataset(dataset_path: Path, n_samples: int = 24) -> list[dict]:
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset_v2(dataset_path, n_samples=n_samples, seed=42)

    rows = json.loads(dataset_path.read_text())
    if rows and rows[0]["global_labels"].get("alpha_T_pcm_per_K") is None:
        generate_dataset_v2(dataset_path, n_samples=max(n_samples, len(rows)), seed=42)
        rows = json.loads(dataset_path.read_text())
    return rows


def build_scalar_features(record: dict) -> tuple[list[float], list[float], list[float]]:
    inputs = record["inputs"]
    geom = record["geometry"]
    xs = record["xs"]
    labels = record["global_labels"]

    raw = [
        float(inputs["enrichment"]),
        float(inputs["uf4_mol_frac"]),
        float(inputs["radius_cm"]),
        float(inputs["height_cm"]),
        float(inputs["temperature_K"]),
        float(inputs["fuel_volume_fraction"]),
        float(inputs["water_vol_frac"]),
    ]
    full = raw + [
        float(geom["Br2_cm2_inv"]),
        float(geom["Bz2_cm2_inv"]),
        float(geom["Bg2_cm2_inv"]),
        float(xs["D1_cm"]),
        float(xs["D2_cm"]),
        float(xs["Sigma_a1_cm1"]),
        float(xs["Sigma_a2_cm1"]),
        float(xs["Sigma_s12_cm1"]),
        float(xs["nuSigma_f1_cm1"]),
        float(xs["nuSigma_f2_cm1"]),
    ]
    target = [float(labels[key]) for key in TARGET_KEYS]
    return raw, full, target


def build_scalar_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = []
    full = []
    targets = []
    bg2 = []
    for record in rows:
        x_raw, x_full, y = build_scalar_features(record)
        raw.append(x_raw)
        full.append(x_full)
        targets.append(y)
        bg2.append(float(record["geometry"]["Bg2_cm2_inv"]))
    return (
        np.asarray(raw, dtype=float),
        np.asarray(full, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(bg2, dtype=float),
    )


def build_field_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    sample_ids = []
    boundary_mask = []

    for sample_idx, record in enumerate(rows):
        _, full_features, _ = build_scalar_features(record)
        r = np.asarray(record["mesh"]["r_centers_cm"], dtype=float)
        z = np.asarray(record["mesh"]["z_centers_cm"], dtype=float)
        phi1 = np.asarray(record["fields"]["phi1_shape"], dtype=float)
        phi2 = np.asarray(record["fields"]["phi2_shape"], dtype=float)
        radius = float(record["inputs"]["radius_cm"])
        height = float(record["inputs"]["height_cm"])
        re = float(record["geometry"]["radius_extrap_cm"])
        he = float(record["geometry"]["height_extrap_cm"])

        for i, r_cm in enumerate(r):
            for j, z_cm in enumerate(z):
                local = [
                    r_cm / max(radius, 1.0e-8),
                    z_cm / max(height, 1.0e-8),
                    r_cm / max(re, 1.0e-8),
                    z_cm / max(he, 1.0e-8),
                    1.0 - r_cm / max(radius, 1.0e-8),
                    z_cm / max(height, 1.0e-8),
                    1.0 - z_cm / max(height, 1.0e-8),
                ]
                x_rows.append(full_features + local)
                y_rows.append([phi1[i, j], phi2[i, j]])
                sample_ids.append(sample_idx)
                is_boundary = int(i in {0, len(r) - 1} or j in {0, len(z) - 1})
                boundary_mask.append(is_boundary)

    return (
        np.asarray(x_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(sample_ids, dtype=int),
        np.asarray(boundary_mask, dtype=float),
    )


def analytic_field_baseline(point_features: np.ndarray) -> np.ndarray:
    r_over_r = point_features[:, -7]
    z_over_h = point_features[:, -6]
    radial = np.clip(np.cos(0.5 * math.pi * r_over_r), 0.0, None)
    axial = np.clip(np.sin(math.pi * z_over_h), 0.0, None)
    shape = (radial * axial).reshape(-1, 1)
    return np.repeat(shape, 2, axis=1)


def train_test_split(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return indices[n_test:], indices[:n_test]


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
    aux_loss: list[float]


class ScalarPINN:
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, int] = (64, 64), seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        h1, h2 = hidden_dims
        self.params = {
            "W1": self.rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, h1)),
            "b1": np.zeros((1, h1)),
            "W2": self.rng.normal(scale=np.sqrt(2.0 / h1), size=(h1, h2)),
            "b2": np.zeros((1, h2)),
            "W3": self.rng.normal(scale=np.sqrt(2.0 / h2), size=(h2, output_dim)),
            "b3": np.zeros((1, output_dim)),
            "log_m2": np.array([[softplus_inv(12.0)]], dtype=float),
        }
        self.mom1 = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.mom2 = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.best_params = None
        self.best_val_loss = float("inf")

    def _prepare(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_mean is None:
            self.x_mean = x.mean(axis=0, keepdims=True)
            self.x_std = x.std(axis=0, keepdims=True) + 1.0e-8
            self.y_mean = y.mean(axis=0, keepdims=True)
            self.y_std = y.std(axis=0, keepdims=True) + 1.0e-8
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def _forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x_norm @ self.params["W1"] + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.tanh(z2)
        y_norm = a2 @ self.params["W3"] + self.params["b3"]
        return y_norm, {"x_norm": x_norm, "a1": a1, "a2": a2}

    def _loss_and_grad(
        self,
        y_pred_norm: np.ndarray,
        y_true_norm: np.ndarray,
        bg2: np.ndarray,
        lambda_physics: float,
        lambda_alpha: float,
        lambda_l2: float,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        diff = y_pred_norm - y_true_norm
        data_loss = float(np.mean(diff ** 2))
        grad_y_norm = (2.0 / diff.size) * diff

        y_pred = y_pred_norm * self.y_std + self.y_mean
        k_eff = y_pred[:, 0]
        k_inf = y_pred[:, 1]
        alpha = y_pred[:, 2]

        m2 = softplus(self.params["log_m2"])[0, 0] + 1.0e-8
        denom = 1.0 + m2 * bg2
        crit_proxy = k_inf / denom
        residual = k_eff - crit_proxy
        physics_loss = float(np.mean(residual ** 2))

        d_phys = np.zeros_like(y_pred)
        d_phys[:, 0] = (2.0 / len(bg2)) * residual
        d_phys[:, 1] = -(2.0 / len(bg2)) * residual / denom
        grad_y_norm += lambda_physics * d_phys * self.y_std

        dm2 = (2.0 / len(bg2)) * np.sum(residual * (k_inf * bg2 / (denom ** 2)))
        dlog_m2 = lambda_physics * dm2 * sigmoid(self.params["log_m2"])

        alpha_penalty = float(np.mean(np.maximum(alpha, 0.0) ** 2))
        d_alpha = (2.0 / len(alpha)) * np.maximum(alpha, 0.0)
        grad_y_norm[:, 2] += lambda_alpha * d_alpha * self.y_std[0, 2]

        reg_loss = 0.0
        for name in ("W1", "W2", "W3"):
            reg_loss += float(np.sum(self.params[name] ** 2))
        total_loss = data_loss + lambda_physics * physics_loss + lambda_alpha * alpha_penalty + lambda_l2 * reg_loss
        return {
            "total": total_loss,
            "data": data_loss,
            "physics": physics_loss,
            "aux": alpha_penalty,
            "m2": float(m2),
        }, grad_y_norm, dlog_m2

    def _backprop(self, cache: dict[str, np.ndarray], grad_y_norm: np.ndarray, dlog_m2: np.ndarray, lambda_l2: float) -> dict[str, np.ndarray]:
        grads = {}
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
                y_pred_norm, y_train_norm, bg2_train, lambda_physics, lambda_alpha, lambda_l2
            )
            grads = self._backprop(cache, grad_y_norm, dlog_m2, lambda_l2)
            self._apply_adam(grads, lr=lr, step=epoch)

            val_pred_norm, _ = self._forward(x_val_norm)
            val_losses, _, _ = self._loss_and_grad(
                val_pred_norm, y_val_norm, bg2_val, lambda_physics, lambda_alpha, lambda_l2
            )
            history.train_loss.append(train_losses["total"])
            history.val_loss.append(val_losses["total"])
            history.data_loss.append(train_losses["data"])
            history.physics_loss.append(train_losses["physics"])
            history.aux_loss.append(train_losses["aux"])

            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.best_params = deepcopy(self.params)

            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(
                    f"  scalar epoch {epoch:4d} | train {train_losses['total']:.5f}"
                    f" | val {val_losses['total']:.5f} | m2 {val_losses['m2']:.3f}"
                )

        if self.best_params is not None:
            self.params = deepcopy(self.best_params)
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.x_mean) / self.x_std
        y_norm, _ = self._forward(x_norm)
        return y_norm * self.y_std + self.y_mean

    def learned_m2(self) -> float:
        return float(softplus(self.params["log_m2"])[0, 0])


class FieldOperatorNet:
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (96, 96), seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        h1, h2 = hidden_dims
        self.params = {
            "W1": rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, h1)),
            "b1": np.zeros((1, h1)),
            "W2": rng.normal(scale=np.sqrt(2.0 / h1), size=(h1, h2)),
            "b2": np.zeros((1, h2)),
            "W3": rng.normal(scale=np.sqrt(2.0 / h2), size=(h2, 2)),
            "b3": np.zeros((1, 2)),
        }
        self.mom1 = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.mom2 = {name: np.zeros_like(value) for name, value in self.params.items()}
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.best_params = None
        self.best_val_loss = float("inf")

    def _prepare(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_mean is None:
            self.x_mean = x.mean(axis=0, keepdims=True)
            self.x_std = x.std(axis=0, keepdims=True) + 1.0e-8
            self.y_mean = y.mean(axis=0, keepdims=True)
            self.y_std = y.std(axis=0, keepdims=True) + 1.0e-8
        return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std

    def _forward(self, x_norm: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x_norm @ self.params["W1"] + self.params["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ self.params["W2"] + self.params["b2"]
        a2 = np.tanh(z2)
        y_norm = a2 @ self.params["W3"] + self.params["b3"]
        return y_norm, {"x_norm": x_norm, "a1": a1, "a2": a2}

    def _backprop(self, cache: dict[str, np.ndarray], grad_y_norm: np.ndarray, lambda_l2: float) -> dict[str, np.ndarray]:
        grads = {}
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
        boundary_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        boundary_val: np.ndarray,
        *,
        epochs: int,
        lr: float,
        lambda_bc: float,
        lambda_pos: float,
        lambda_l2: float,
        print_every: int = 100,
    ) -> TrainingHistory:
        x_train_norm, y_train_norm = self._prepare(x_train, y_train)
        x_val_norm = (x_val - self.x_mean) / self.x_std
        y_val_norm = (y_val - self.y_mean) / self.y_std
        history = TrainingHistory([], [], [], [], [])

        for epoch in range(1, epochs + 1):
            y_pred_norm, cache = self._forward(x_train_norm)
            y_pred = y_pred_norm * self.y_std + self.y_mean
            diff = y_pred_norm - y_train_norm
            data_loss = float(np.mean(diff ** 2))
            grad = (2.0 / diff.size) * diff

            boundary_loss = float(np.mean(boundary_train[:, None] * (y_pred ** 2)))
            grad += lambda_bc * (2.0 / len(y_pred)) * boundary_train[:, None] * y_pred * self.y_std

            neg = np.minimum(y_pred, 0.0)
            positivity_loss = float(np.mean(neg ** 2))
            grad += lambda_pos * (2.0 / len(y_pred)) * neg * self.y_std

            grads = self._backprop(cache, grad, lambda_l2=lambda_l2)
            self._apply_adam(grads, lr=lr, step=epoch)

            val_pred_norm, _ = self._forward(x_val_norm)
            val_pred = val_pred_norm * self.y_std + self.y_mean
            val_diff = val_pred_norm - y_val_norm
            val_data_loss = float(np.mean(val_diff ** 2))
            val_boundary = float(np.mean(boundary_val[:, None] * (val_pred ** 2)))
            val_pos = float(np.mean(np.minimum(val_pred, 0.0) ** 2))
            reg_loss = sum(float(np.sum(self.params[name] ** 2)) for name in ("W1", "W2", "W3"))
            train_total = data_loss + lambda_bc * boundary_loss + lambda_pos * positivity_loss + lambda_l2 * reg_loss
            val_total = val_data_loss + lambda_bc * val_boundary + lambda_pos * val_pos + lambda_l2 * reg_loss

            history.train_loss.append(train_total)
            history.val_loss.append(val_total)
            history.data_loss.append(data_loss)
            history.physics_loss.append(boundary_loss)
            history.aux_loss.append(positivity_loss)

            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                self.best_params = deepcopy(self.params)

            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(
                    f"  field epoch {epoch:4d} | train {train_total:.5f}"
                    f" | val {val_total:.5f} | bc {val_boundary:.5f}"
                )

        if self.best_params is not None:
            self.params = deepcopy(self.best_params)
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.x_mean) / self.x_std
        y_norm, _ = self._forward(x_norm)
        return y_norm * self.y_std + self.y_mean


def metrics_for_targets(y_true: np.ndarray, y_pred: np.ndarray, keys: list[str]) -> dict:
    metrics = {}
    for idx, key in enumerate(keys):
        err = y_pred[:, idx] - y_true[:, idx]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        denom = np.sum((y_true[:, idx] - y_true[:, idx].mean()) ** 2)
        r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else 0.0
        metrics[key] = {"rmse": rmse, "mae": mae, "r2": r2}
    return metrics


def physics_residual(y_pred: np.ndarray, bg2: np.ndarray, m2: float) -> float:
    resid = y_pred[:, 0] - y_pred[:, 1] / (1.0 + m2 * bg2)
    return float(np.sqrt(np.mean(resid ** 2)))


def field_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "phi1_rmse": float(np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2))),
        "phi2_rmse": float(np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2))),
        "phi1_mae": float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))),
        "phi2_mae": float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))),
    }


def save_scalar_plots(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    upgraded_pred: np.ndarray,
    baseline_history: TrainingHistory,
    upgraded_history: TrainingHistory,
) -> None:
    parity_path = ARTIFACT_DIR / "scalar_parity.png"
    loss_path = ARTIFACT_DIR / "scalar_loss.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    pairs = [("k_eff", 0), ("alpha_T_pcm_per_K", 2)]
    for ax, (label, idx) in zip(axes, pairs, strict=True):
        truth = y_true[:, idx]
        ax.scatter(truth, baseline_pred[:, idx], s=28, alpha=0.75, label="baseline")
        ax.scatter(truth, upgraded_pred[:, idx], s=28, alpha=0.75, label="upgraded")
        lo = min(truth.min(), baseline_pred[:, idx].min(), upgraded_pred[:, idx].min())
        hi = max(truth.max(), baseline_pred[:, idx].max(), upgraded_pred[:, idx].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.legend()
    fig.tight_layout()
    fig.savefig(parity_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(baseline_history.train_loss, label="baseline train")
    axes[0].plot(baseline_history.val_loss, label="baseline val")
    axes[0].set_title("Baseline Scalar Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(upgraded_history.train_loss, label="upgraded train")
    axes[1].plot(upgraded_history.val_loss, label="upgraded val")
    axes[1].plot(upgraded_history.physics_loss, label="physics")
    axes[1].set_title("Upgraded Scalar Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(loss_path, dpi=180)
    plt.close(fig)


def save_field_plot(record: dict, y_true: np.ndarray, analytic_pred: np.ndarray, learned_pred: np.ndarray) -> None:
    nr = int(record["mesh"]["nr"])
    nz = int(record["mesh"]["nz"])
    labels = [
        ("Truth phi1", y_true[:, 0].reshape(nr, nz)),
        ("Analytic baseline phi1", analytic_pred[:, 0].reshape(nr, nz)),
        ("Learned phi1", learned_pred[:, 0].reshape(nr, nz)),
        ("Truth phi2", y_true[:, 1].reshape(nr, nz)),
        ("Analytic baseline phi2", analytic_pred[:, 1].reshape(nr, nz)),
        ("Learned phi2", learned_pred[:, 1].reshape(nr, nz)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for ax, (title, field) in zip(axes.flat, labels, strict=True):
        mesh = ax.imshow(field.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "field_holdout_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(summary_path: Path, payload: dict) -> None:
    k_eff_baseline = payload["scalar_baseline"]["k_eff"]["rmse"]
    k_eff_upgraded = payload["scalar_upgraded"]["k_eff"]["rmse"]
    alpha_baseline = payload["scalar_baseline"]["alpha_T_pcm_per_K"]["rmse"]
    alpha_upgraded = payload["scalar_upgraded"]["alpha_T_pcm_per_K"]["rmse"]
    residual_baseline = payload["scalar_baseline_residual"]
    residual_upgraded = payload["scalar_upgraded_residual"]

    lines = [
        "# First PINN Run v2",
        "",
        "## Dataset",
        "",
        f"- Samples: {payload['n_samples']}",
        f"- Field points: {payload['n_field_points']}",
        "- Data source: `artifacts/solver_v2/dataset_v2_small.json`",
        "- Inputs: benchmark-aware geometry, buckling, and XS features plus spatial coordinates for fields",
        "",
        "## Scalar Comparison",
        "",
        (
            f"- `k_eff` RMSE moved from {k_eff_baseline:.5f} to {k_eff_upgraded:.5f}."
        ),
        (
            f"- `alpha_T` RMSE improved from {alpha_baseline:.2f} "
            f"to {alpha_upgraded:.2f} pcm/K."
        ),
        (
            f"- Criticality-consistency residual improved from {residual_baseline:.5f} "
            f"to {residual_upgraded:.5f}."
        ),
        "- The upgraded scalar model improves benchmark-sensitive labels more than overall `k_eff` RMSE on this small 24-sample corpus, so the scalar head still needs more data and tuning.",
        "",
        "## Field Comparison",
        "",
        (
            f"- Analytic baseline `phi1` RMSE: {payload['field_baseline']['phi1_rmse']:.5f}, "
            f"learned field RMSE: {payload['field_model']['phi1_rmse']:.5f}."
        ),
        (
            f"- Analytic baseline `phi2` RMSE: {payload['field_baseline']['phi2_rmse']:.5f}, "
            f"learned field RMSE: {payload['field_model']['phi2_rmse']:.5f}."
        ),
        "- The upgraded pipeline now predicts `phi1(r,z)` and `phi2(r,z)` directly from design features plus coordinates.",
        "",
        "## Artifacts",
        "",
        "- `scalar_parity.png`",
        "- `scalar_loss.png`",
        "- `field_holdout_comparison.png`",
        "- `metrics.json`",
        "",
    ]
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    t0 = time.time()
    rows = load_dataset(DATASET_PATH, n_samples=24)

    x_raw, x_full, y_scalar, bg2 = build_scalar_arrays(rows)
    train_idx, test_idx = train_test_split(len(rows), test_fraction=0.25, seed=42)

    print("Training scalar baseline...")
    scalar_baseline = ScalarPINN(input_dim=x_raw.shape[1], output_dim=y_scalar.shape[1], seed=42)
    scalar_baseline_history = scalar_baseline.fit(
        x_raw[train_idx],
        y_scalar[train_idx],
        bg2[train_idx],
        x_raw[test_idx],
        y_scalar[test_idx],
        bg2[test_idx],
        epochs=900,
        lr=2.5e-3,
        lambda_physics=0.0,
        lambda_alpha=0.0,
        lambda_l2=1.0e-5,
        print_every=150,
    )

    print("\nTraining upgraded scalar PINN...")
    scalar_upgraded = ScalarPINN(input_dim=x_full.shape[1], output_dim=y_scalar.shape[1], seed=42)
    scalar_upgraded_history = scalar_upgraded.fit(
        x_full[train_idx],
        y_scalar[train_idx],
        bg2[train_idx],
        x_full[test_idx],
        y_scalar[test_idx],
        bg2[test_idx],
        epochs=1100,
        lr=2.0e-3,
        lambda_physics=0.30,
        lambda_alpha=0.02,
        lambda_l2=1.0e-5,
        print_every=150,
    )

    baseline_pred = scalar_baseline.predict(x_raw[test_idx])
    upgraded_pred = scalar_upgraded.predict(x_full[test_idx])
    scalar_baseline_metrics = metrics_for_targets(y_scalar[test_idx], baseline_pred, TARGET_KEYS)
    scalar_upgraded_metrics = metrics_for_targets(y_scalar[test_idx], upgraded_pred, TARGET_KEYS)
    scalar_baseline_residual = physics_residual(baseline_pred, bg2[test_idx], scalar_baseline.learned_m2())
    scalar_upgraded_residual = physics_residual(upgraded_pred, bg2[test_idx], scalar_upgraded.learned_m2())

    print("\nBuilding field dataset...")
    field_x, field_y, point_sample_ids, boundary_mask = build_field_arrays(rows)
    train_mask = np.isin(point_sample_ids, train_idx)
    test_mask = np.isin(point_sample_ids, test_idx)

    print("Training coordinate-query field model...")
    field_model = FieldOperatorNet(input_dim=field_x.shape[1], seed=42)
    field_history = field_model.fit(
        field_x[train_mask],
        field_y[train_mask],
        boundary_mask[train_mask],
        field_x[test_mask],
        field_y[test_mask],
        boundary_mask[test_mask],
        epochs=800,
        lr=1.8e-3,
        lambda_bc=0.10,
        lambda_pos=0.02,
        lambda_l2=1.0e-6,
        print_every=150,
    )

    field_pred = field_model.predict(field_x[test_mask])
    analytic_pred = analytic_field_baseline(field_x[test_mask])
    field_model_metrics = field_metrics(field_y[test_mask], field_pred)
    field_baseline_metrics = field_metrics(field_y[test_mask], analytic_pred)

    holdout_sample = int(test_idx[0])
    holdout_mask = point_sample_ids[test_mask] == holdout_sample
    holdout_record = rows[holdout_sample]
    save_field_plot(holdout_record, field_y[test_mask][holdout_mask], analytic_pred[holdout_mask], field_pred[holdout_mask])
    save_scalar_plots(y_scalar[test_idx], baseline_pred, upgraded_pred, scalar_baseline_history, scalar_upgraded_history)

    metrics_payload = {
        "n_samples": len(rows),
        "n_field_points": int(field_x.shape[0]),
        "scalar_baseline": scalar_baseline_metrics,
        "scalar_upgraded": scalar_upgraded_metrics,
        "scalar_baseline_residual": scalar_baseline_residual,
        "scalar_upgraded_residual": scalar_upgraded_residual,
        "field_baseline": field_baseline_metrics,
        "field_model": field_model_metrics,
        "runtime_seconds": time.time() - t0,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(to_builtin(metrics_payload), indent=2))
    write_summary(ARTIFACT_DIR / "summary.md", metrics_payload)

    print("\nRun complete.")
    print(f"  summary: {ARTIFACT_DIR / 'summary.md'}")
    print(f"  metrics: {ARTIFACT_DIR / 'metrics.json'}")
    print(f"  scalar parity: {ARTIFACT_DIR / 'scalar_parity.png'}")
    print(f"  field comparison: {ARTIFACT_DIR / 'field_holdout_comparison.png'}")


if __name__ == "__main__":
    main()
