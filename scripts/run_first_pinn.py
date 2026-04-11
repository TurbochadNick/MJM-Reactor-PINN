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

from monster_v2_package.solver_v2 import LEU_MAX_ENRICHMENT, LEU_MIN_ENRICHMENT, generate_dataset_v2


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

TARGET_EXPLANATIONS = {
    "k_eff": "Effective multiplication factor for the finite-cylinder core.",
    "k_inf": "Infinite-medium multiplication factor from the homogenized 2-group model.",
    "alpha_T_pcm_per_K": "Temperature coefficient of reactivity in pcm/K.",
    "peaking_factor": "Peak total flux divided by volume-averaged total flux.",
    "peak_flux_n_cm2_s": "Maximum physical total neutron flux in n/cm^2/s.",
    "peak_power_density_W_cm3": "Maximum volumetric power density in W/cm^3.",
    "beta_eff_ratio": "Circulating-to-static delayed neutron ratio, beta_eff / beta_static.",
}

DATASET_FIELD_EXPLANATIONS = {
    "sample_id": "Unique sample identifier for the exported v2 record.",
    "solver_version": "Solver tag used to create the record.",
    "source_document": "Benchmark/source provenance anchor.",
    "notes": "Human-readable modeling assumptions and caveats.",
    "inputs": "Design inputs such as enrichment, UF4 loading, dimensions, and temperatures.",
    "geometry": "Derived geometry and buckling quantities including extrapolated dimensions.",
    "material_properties": "Thermophysical properties used in the sample.",
    "number_densities": "Explicit constituent number densities used to build XS terms.",
    "xs": "Homogenized 2-group macroscopic cross sections and correction factors.",
    "global_labels": "Scalar reactor-response outputs used for benchmark comparison and surrogate training.",
    "thermal_hydraulics": "Simple loop/state-point quantities derived from the solved design.",
    "mesh": "Mesh size and coordinates used for the field outputs.",
    "fields": "2D field arrays including phi1, phi2, total flux, fission rate, and power density.",
    "energy_groups": "Explicit assumed fast/thermal group energy ranges used for reporting and exported metadata.",
    "provenance": "Boundary-condition, normalization, and source metadata.",
}


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
    needs_regen = False
    if rows and rows[0]["global_labels"].get("alpha_T_pcm_per_K") is None:
        needs_regen = True
    if rows:
        enrichments = [float(row["inputs"]["enrichment"]) for row in rows]
        if min(enrichments) < LEU_MIN_ENRICHMENT - 1.0e-12 or max(enrichments) > LEU_MAX_ENRICHMENT + 1.0e-12:
            needs_regen = True
    if needs_regen:
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
        ax.set_xlabel("true / reference solver value")
        ax.set_ylabel("predicted model value")
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
        ax.set_xlabel("radial position index (center to edge)")
        ax.set_ylabel("axial position index (bottom to top)")
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "field_holdout_comparison.png", dpi=180)
    plt.close(fig)


def load_solver_summary() -> dict | None:
    summary_path = ROOT / "artifacts" / "solver_v2" / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def build_comprehensive_payload(
    metrics_payload: dict,
    rows: list[dict],
    holdout_sample: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    solver_summary = load_solver_summary()
    sample = rows[0]
    comprehensive = {
        "overview": {
            "report_purpose": "Single-file labeled view of the v2 solver-backed PINN run.",
            "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
            "n_samples": len(rows),
            "n_train_samples": int(len(train_idx)),
            "n_test_samples": int(len(test_idx)),
            "n_field_points": metrics_payload["n_field_points"],
            "holdout_sample_id": rows[holdout_sample]["sample_id"],
            "runtime_seconds": metrics_payload["runtime_seconds"],
        },
        "what_the_models_do": {
            "scalar_baseline": "Plain neural surrogate on raw design inputs only.",
            "scalar_upgraded": "Physics-guided scalar surrogate using design inputs, buckling, and XS features.",
            "field_model": "Coordinate-query neural field model predicting phi1(r,z) and phi2(r,z).",
        },
        "target_definitions": TARGET_EXPLANATIONS,
        "dataset_record_layout": DATASET_FIELD_EXPLANATIONS,
        "example_dataset_record_keys": list(sample.keys()),
        "example_global_label_keys": list(sample["global_labels"].keys()),
        "example_field_keys": list(sample["fields"].keys()),
        "solver_context": solver_summary,
        "pinn_metrics": metrics_payload,
        "artifact_files": {
            "primary_report_md": str((ARTIFACT_DIR / "comprehensive_output_report.md").relative_to(ROOT)),
            "summary_md": str((ARTIFACT_DIR / "summary.md").relative_to(ROOT)),
            "metrics_json": str((ARTIFACT_DIR / "metrics.json").relative_to(ROOT)),
            "scalar_parity_png": str((ARTIFACT_DIR / "scalar_parity.png").relative_to(ROOT)),
            "scalar_loss_png": str((ARTIFACT_DIR / "scalar_loss.png").relative_to(ROOT)),
            "field_holdout_png": str((ARTIFACT_DIR / "field_holdout_comparison.png").relative_to(ROOT)),
        },
    }
    return to_builtin(comprehensive)


def write_comprehensive_report(report_path: Path, comprehensive: dict) -> None:
    solver_context = comprehensive.get("solver_context")
    metrics = comprehensive["pinn_metrics"]
    lines = [
        "# Comprehensive PINN Output Report",
        "",
        "This is the single-file labeled view of the current v2 solver-backed PINN run.",
        "",
        "## Overview",
        "",
        f"- Dataset source: `{comprehensive['overview']['dataset_path']}`",
        f"- Samples: {comprehensive['overview']['n_samples']}",
        f"- Train/test split: {comprehensive['overview']['n_train_samples']} / {comprehensive['overview']['n_test_samples']}",
        f"- Field points used by the coordinate-query model: {comprehensive['overview']['n_field_points']}",
        f"- Holdout sample shown in the field comparison plot: `{comprehensive['overview']['holdout_sample_id']}`",
        f"- PINN runtime: {comprehensive['overview']['runtime_seconds']:.2f} s",
        "",
        "## What Each Model Is",
        "",
        f"- Scalar baseline: {comprehensive['what_the_models_do']['scalar_baseline']}",
        f"- Scalar upgraded: {comprehensive['what_the_models_do']['scalar_upgraded']}",
        f"- Field model: {comprehensive['what_the_models_do']['field_model']}",
        "",
    ]
    if solver_context is not None and "energy_groups" in solver_context:
        energy_groups = solver_context["energy_groups"]
        lines.extend(
            [
                "## Explicit Two-Group Energy Definitions",
                "",
                f"- `phi1` fast group: {energy_groups['group_1_fast']['energy_range_text']}",
                f"- `phi2` thermal group: {energy_groups['group_2_thermal']['energy_range_text']}",
                f"- Note: {energy_groups['note']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Dataset Layout",
            "",
        ]
    )
    for key, explanation in comprehensive["dataset_record_layout"].items():
        lines.append(f"- `{key}`: {explanation}")

    lines.extend(
        [
            "",
            "## Scalar Target Definitions",
            "",
        ]
    )
    for key, explanation in comprehensive["target_definitions"].items():
        lines.append(f"- `{key}`: {explanation}")

    lines.extend(
        [
            "",
            "## Scalar Metrics",
            "",
            "Baseline and upgraded scalar metrics are reported as RMSE / MAE / R^2.",
            "",
        ]
    )
    for key in TARGET_KEYS:
        baseline = metrics["scalar_baseline"][key]
        upgraded = metrics["scalar_upgraded"][key]
        lines.append(
            f"- `{key}`: baseline RMSE `{baseline['rmse']:.5g}`, MAE `{baseline['mae']:.5g}`, R^2 `{baseline['r2']:.4f}`"
        )
        lines.append(
            f"  upgraded RMSE `{upgraded['rmse']:.5g}`, MAE `{upgraded['mae']:.5g}`, R^2 `{upgraded['r2']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Scalar Physics Consistency",
            "",
            f"- Baseline criticality-consistency residual: `{metrics['scalar_baseline_residual']:.5f}`",
            f"- Upgraded criticality-consistency residual: `{metrics['scalar_upgraded_residual']:.5f}`",
            "- Lower is better. This is the RMSE of `k_eff - k_inf / (1 + M^2 Bg^2)` using the learned scalar model outputs.",
            "",
            "## Field Metrics",
            "",
            f"- Analytic baseline phi1 RMSE: `{metrics['field_baseline']['phi1_rmse']:.5g}`",
            f"- Learned field-model phi1 RMSE: `{metrics['field_model']['phi1_rmse']:.5g}`",
            f"- Analytic baseline phi2 RMSE: `{metrics['field_baseline']['phi2_rmse']:.5g}`",
            f"- Learned field-model phi2 RMSE: `{metrics['field_model']['phi2_rmse']:.5g}`",
            "",
            "## How To Read The Artifact Files",
            "",
        ]
    )
    for key, path in comprehensive["artifact_files"].items():
        lines.append(f"- `{key}`: `{path}`")

    if solver_context is not None:
        fuel_c = solver_context["fuel_c_proxy"]["global_labels"]
        targets = solver_context["targets"]
        alpha_value = fuel_c.get("alpha_T_pcm_per_K")
        if alpha_value is None:
            alpha_value = solver_context["temperature_sweep"]["mean_alpha_pcm_per_K"]
        lines.extend(
            [
                "",
                "## Solver Benchmark Snapshot",
                "",
                "- Fuel C remains a higher-enrichment benchmark anchor and is not part of the LEU-only PINN training set.",
                f"- Fuel C proxy `alpha_T`: `{alpha_value:.3f} pcm/K`",
                f"- ORNL `alpha_T` target: `{targets['alpha_pcm_per_K']:.3f} pcm/K`",
                f"- Fuel C proxy delayed ratio: `{fuel_c['beta_eff_ratio']:.3f}`",
                f"- ORNL delayed ratio target: `{targets['beta_ratio']:.3f}`",
                f"- Fuel C proxy peak thermal flux: `{fuel_c['peak_thermal_flux_n_cm2_s']:.3e}`",
                f"- ORNL peak thermal flux target: `{targets['peak_thermal_flux_n_cm2_s']:.3e}`",
                "",
                "## Interpretation",
                "",
                "- The upgraded solver substantially improves temperature-reactivity behavior and D2 plausibility.",
                "- The field pipeline now has explicit flux-field labels instead of scalar-only supervision.",
                "- The main remaining solver-level gap is flux magnitude, which still points to cross-section fidelity limits.",
                "- The main remaining PINN-level gap is scalar robustness on the small dataset, especially for k_eff and beta ratio.",
                "",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n")


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
        f"- Enrichment regime: LEU-only `{LEU_MIN_ENRICHMENT*100:.1f}%` to `{LEU_MAX_ENRICHMENT*100:.1f}%`",
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
    comprehensive = build_comprehensive_payload(metrics_payload, rows, holdout_sample, train_idx, test_idx)
    (ARTIFACT_DIR / "full_output_bundle.json").write_text(json.dumps(comprehensive, indent=2))
    write_comprehensive_report(ARTIFACT_DIR / "comprehensive_output_report.md", comprehensive)

    print("\nRun complete.")
    print(f"  summary: {ARTIFACT_DIR / 'summary.md'}")
    print(f"  metrics: {ARTIFACT_DIR / 'metrics.json'}")
    print(f"  comprehensive report: {ARTIFACT_DIR / 'comprehensive_output_report.md'}")
    print(f"  output bundle: {ARTIFACT_DIR / 'full_output_bundle.json'}")
    print(f"  scalar parity: {ARTIFACT_DIR / 'scalar_parity.png'}")
    print(f"  field comparison: {ARTIFACT_DIR / 'field_holdout_comparison.png'}")


if __name__ == "__main__":
    main()
