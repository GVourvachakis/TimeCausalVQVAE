"""Discrete-latent geometry diagnostics for VQ-family tokenizers.

References
----------
- Neural Discrete Representation Learning, van den Oord, Vinyals, and Kavukcuoglu
(DOI: 10.5555/3295222.3295378) - motivates inspection of learned discrete codebooks.

- vector-quantize-pytorch, lucidrains
(repository: https://github.com/lucidrains/vector-quantize-pytorch) - source of wrapped VQ-family
backends whose codebooks are inspected here.

- MGVQ: Could VQ-VAE Beat VAE? A Generalizable Tokenizer with Multi-group Quantization
(arXiv DOI: 10.48550/arXiv.2507.07997) - future grouped-tokenizer motivation only.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from time_causal_vae.evaluation.style import apply_clean_style


@dataclass(frozen=True)
class CodebookGeometry:
    """Flattened codebook vectors with component metadata."""

    embeddings: np.ndarray
    code_indices: list[int]
    quantizer_indices: list[int | None]
    group_indices: list[int | None]
    labels: list[str]
    source: str
    notes: list[str]


@dataclass(frozen=True)
class TokenArtifactData:
    """Combined token artifact tensors used by geometry diagnostics."""

    indices: Tensor
    labels: Tensor | None
    data: Tensor | None
    source_files: list[str]


def extract_codebook_geometry(
    tokenizer: nn.Module,
    *,
    observed_indices: Tensor | None,
    codebook_size: int,
    quantizer_type: str,
    num_quantizers: int = 1,
    groups: int = 1,
) -> CodebookGeometry:
    """Extract codebook embeddings, falling back to observed quantized embeddings."""
    notes: list[str] = []
    direct = _direct_codebook_from_tokenizer(tokenizer)
    if direct is not None:
        return geometry_from_array(
            direct,
            quantizer_type=quantizer_type,
            source="direct_backend_state",
            notes=notes,
        )
    notes.append("Direct backend codebook extraction was unavailable.")
    decoded = _decode_all_codes(
        tokenizer, codebook_size=codebook_size, quantizer_type=quantizer_type
    )
    if decoded is not None:
        return geometry_from_array(
            decoded,
            quantizer_type=quantizer_type,
            source="decoded_single_code_probe",
            notes=notes,
        )
    if observed_indices is None:
        raise ValueError(
            "Observed indices are required when codebook extraction is unavailable.")
    notes.append(
        "Falling back to mean decoded embeddings from observed token indices.")
    return observed_quantized_geometry(
        tokenizer,
        observed_indices=observed_indices,
        codebook_size=codebook_size,
        quantizer_type=quantizer_type,
        num_quantizers=num_quantizers,
        groups=groups,
        notes=notes,
    )


def geometry_from_array(
    embeddings: np.ndarray | Tensor,
    *,
    quantizer_type: str,
    source: str,
    notes: Sequence[str] = (),
) -> CodebookGeometry:
    """Convert direct codebook tensors to a flattened geometry table."""
    array = to_numpy(embeddings)
    array = np.squeeze(array)
    if array.ndim == 2:
        flat = array
        code_indices = list(range(flat.shape[0]))
        quantizer_indices: list[int | None] = [None] * flat.shape[0]
        group_indices: list[int | None] = [None] * flat.shape[0]
    elif array.ndim == 3:
        flat = array.reshape(array.shape[0] * array.shape[1], array.shape[2])
        code_indices = [
            code for _quantizer in range(array.shape[0]) for code in range(array.shape[1])
        ]
        quantizer_indices = [
            quantizer for quantizer in range(array.shape[0]) for _code in range(array.shape[1])
        ]
        group_indices = [None] * flat.shape[0]
    elif array.ndim == 4:
        flat = array.reshape(
            array.shape[0] * array.shape[1] * array.shape[2], array.shape[3])
        code_indices = [
            code
            for _group in range(array.shape[0])
            for _quantizer in range(array.shape[1])
            for code in range(array.shape[2])
        ]
        quantizer_indices = [
            quantizer
            for _group in range(array.shape[0])
            for quantizer in range(array.shape[1])
            for _code in range(array.shape[2])
        ]
        group_indices = [
            group
            for group in range(array.shape[0])
            for _quantizer in range(array.shape[1])
            for _code in range(array.shape[2])
        ]
    else:
        raise ValueError(
            "Codebook embeddings must have rank 2, 3, or 4 after squeezing; "
            f"got shape {array.shape}."
        )
    labels = [
        code_label(code, quantizer=quantizer, group=group)
        for code, quantizer, group in zip(
            code_indices,
            quantizer_indices,
            group_indices,
            strict=True,
        )
    ]
    return CodebookGeometry(
        embeddings=flat.astype(np.float64, copy=False),
        code_indices=code_indices,
        quantizer_indices=quantizer_indices,
        group_indices=group_indices,
        labels=labels,
        source=source,
        notes=[
            *notes,
            f"Interpreted codebook tensor shape {list(array.shape)} for {quantizer_type}.",
        ],
    )


def observed_quantized_geometry(
    tokenizer: nn.Module,
    *,
    observed_indices: Tensor,
    codebook_size: int,
    quantizer_type: str,
    num_quantizers: int,
    groups: int,
    notes: Sequence[str],
) -> CodebookGeometry:
    """Approximate geometry by averaging decoded embeddings for observed indices."""
    with torch.no_grad():
        quantized = cast(Any, tokenizer).quantizer.decode_indices(
            observed_indices).detach().cpu()
    if quantized.shape[:2] != observed_indices.shape[:2]:
        raise ValueError(
            "Decoded quantized embeddings must share the batch/time prefix with indices; "
            f"got {tuple(quantized.shape)} and {tuple(observed_indices.shape)}."
        )
    if observed_indices.ndim == 2:
        return _observed_vector_geometry(quantized, observed_indices, codebook_size, notes=notes)
    if observed_indices.ndim == 3:
        return _observed_rvq_geometry(
            quantized,
            observed_indices,
            codebook_size,
            num_quantizers=num_quantizers,
            notes=notes,
        )
    if observed_indices.ndim == 4:
        return _observed_grouped_geometry(
            quantized,
            observed_indices,
            codebook_size,
            groups=groups,
            num_quantizers=num_quantizers,
            notes=notes,
        )
    raise ValueError(
        "Observed indices must have rank 2, 3, or 4 for VQ-family tokenizers; "
        f"got {tuple(observed_indices.shape)} for {quantizer_type}."
    )


def project_embeddings(embeddings: np.ndarray, *, method: str = "pca") -> tuple[np.ndarray, str]:
    """Project codebook embeddings to two dimensions."""
    if method != "pca":
        raise ValueError(f"Unsupported projection method: {method}")
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected [n_code, dim] embeddings; got {embeddings.shape}.")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot project an empty codebook.")
    centred = embeddings - embeddings.mean(axis=0, keepdims=True)
    if embeddings.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64), "single_point"
    try:
        from sklearn.decomposition import PCA

        projection = PCA(
            n_components=2, random_state=0).fit_transform(embeddings)
        return cast(np.ndarray, projection), "sklearn_pca"
    except Exception:
        _u, _singular, vh = np.linalg.svd(centred, full_matrices=False)
        components = vh[:2].T
        if components.shape[1] == 1:
            projection = np.column_stack(
                [centred @ components[:, 0], np.zeros(embeddings.shape[0])]
            )
        else:
            projection = centred @ components
        return projection.astype(np.float64, copy=False), "numpy_svd_pca"


def code_usage_summary(indices: Tensor, codebook_size: int) -> dict[str, Any]:
    """Summarise marginal and component-level code usage."""
    counts = torch.bincount(indices.detach().cpu().reshape(-1), minlength=codebook_size)[
        :codebook_size
    ]
    summary = summarise_counts(counts)
    if indices.ndim == 2:
        summary["component_usage_note"] = "single_vector_code_per_time_step"
    elif indices.ndim == 3:
        summary["per_quantizer"] = [
            {
                "quantizer_index": quantizer,
                **summarise_counts(
                    torch.bincount(
                        indices[:, :, quantizer].detach().cpu().reshape(-1),
                        minlength=codebook_size,
                    )[:codebook_size]
                ),
            }
            for quantizer in range(indices.shape[2])
        ]
    elif indices.ndim == 4:
        per_group_quantizer = []
        for group in range(indices.shape[2]):
            for quantizer in range(indices.shape[3]):
                per_group_quantizer.append(
                    {
                        "group_index": group,
                        "quantizer_index": quantizer,
                        **summarise_counts(
                            torch.bincount(
                                indices[:, :, group, quantizer].detach(
                                ).cpu().reshape(-1),
                                minlength=codebook_size,
                            )[:codebook_size]
                        ),
                    }
                )
        summary["per_group_quantizer"] = per_group_quantizer
    else:
        summary["component_usage_note"] = (
            f"Component usage omitted for unsupported index rank {indices.ndim}."
        )
    if indices.ndim == 3 and indices.shape[-1] == 2:
        summary["q0_q1_pair"] = q0_q1_pair_summary(indices, codebook_size)
    return summary


def summarise_counts(counts: Tensor) -> dict[str, Any]:
    """Return active count, probabilities, entropy, and perplexity for code counts."""
    counts_cpu = counts.detach().cpu().long()
    total = int(counts_cpu.sum().item())
    active = int((counts_cpu > 0).sum().item())
    codebook_size = int(counts_cpu.numel())
    if total == 0:
        probabilities = torch.zeros_like(counts_cpu, dtype=torch.float32)
        entropy = 0.0
        perplexity = 0.0
        active_indices: list[int] = []
    else:
        probabilities = counts_cpu.float() / float(total)
        active_probabilities = probabilities[probabilities > 0.0]
        entropy_tensor = -(active_probabilities *
                           active_probabilities.log()).sum()
        entropy = float(entropy_tensor.item())
        perplexity = float(torch.exp(entropy_tensor).item())
        active_indices = [
            int(index) for index in torch.nonzero(counts_cpu > 0, as_tuple=False).flatten()
        ]
    return {
        "codebook_size": codebook_size,
        "active_code_count": active,
        "active_code_ratio": active / codebook_size if codebook_size else 0.0,
        "codebook_perplexity": perplexity,
        "index_entropy": entropy,
        "token_count": total,
        "active_code_indices": active_indices,
        "code_usage_counts": [int(value) for value in counts_cpu.tolist()],
        "code_usage_probability": [float(value) for value in probabilities.tolist()],
    }


def condition_bucket_usage(
    *,
    indices: Tensor,
    labels: Tensor | None,
    codebook_size: int,
    n_buckets: int = 5,
) -> list[dict[str, Any]]:
    """Compute code usage by equal-count scalar-condition bucket."""
    if labels is None:
        return []
    values = condition_values(labels)
    if values.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Expected one label per path; got labels {tuple(labels.shape)} and "
            f"indices {tuple(indices.shape)}."
        )
    if values.numel() == 0:
        return []
    bucket_names = ("very_low", "low", "mid", "high", "very_high")
    sorted_positions = torch.argsort(values)
    bucket_count = min(n_buckets, int(values.shape[0]))
    buckets: list[dict[str, Any]] = []
    for bucket_index, positions in enumerate(torch.tensor_split(sorted_positions, bucket_count)):
        if positions.numel() == 0:
            continue
        bucket_indices = indices.index_select(0, positions)
        bucket_values = values.index_select(0, positions)
        usage = code_usage_summary(bucket_indices, codebook_size)
        buckets.append(
            {
                "bucket_index": bucket_index,
                "bucket_label": bucket_names[bucket_index]
                if bucket_index < len(bucket_names)
                else f"bucket_{bucket_index}",
                "n_samples": int(positions.numel()),
                "condition_min": float(bucket_values.min().item()),
                "condition_max": float(bucket_values.max().item()),
                "condition_mean": float(bucket_values.mean().item()),
                "active_code_count": usage["active_code_count"],
                "active_code_ratio": usage["active_code_ratio"],
                "codebook_perplexity": usage["codebook_perplexity"],
                "index_entropy": usage["index_entropy"],
                "code_usage_counts": usage["code_usage_counts"],
                "code_usage_probability": usage["code_usage_probability"],
            }
        )
    return buckets


def q0_q1_pair_summary(indices: Tensor, codebook_size: int) -> dict[str, Any]:
    """Summarise q0/q1 pair usage for two-level RVQ token indices."""
    if indices.ndim != 3 or indices.shape[-1] != 2:
        raise ValueError(
            f"Expected [batch, time, 2] indices; got {tuple(indices.shape)}.")
    pairs = indices.detach().cpu().long().reshape(-1, 2)
    pair_counts = torch.zeros((codebook_size, codebook_size), dtype=torch.long)
    valid = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 0] < codebook_size)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 1] < codebook_size)
    )
    for q0, q1 in pairs[valid].tolist():
        pair_counts[int(q0), int(q1)] += 1
    total_pairs = int(pair_counts.sum().item())
    active_pairs = int((pair_counts > 0).sum().item())
    absent_pair_mass = 1.0 - active_pairs / \
        float(codebook_size * codebook_size)
    return {
        "q0": summarise_counts(
            torch.bincount(indices[:, :, 0].reshape(-1),
                           minlength=codebook_size)[:codebook_size]
        ),
        "q1": summarise_counts(
            torch.bincount(indices[:, :, 1].reshape(-1),
                           minlength=codebook_size)[:codebook_size]
        ),
        "pair_count": total_pairs,
        "active_pair_count": active_pairs,
        "active_pair_ratio": active_pairs / float(codebook_size * codebook_size),
        "absent_pair_mass": absent_pair_mass,
        "pair_counts": [[int(value) for value in row] for row in pair_counts.tolist()],
    }


def load_token_artifacts(token_data_dir: str | Path) -> TokenArtifactData:
    """Load and concatenate train/eval token artifacts from an extraction directory."""
    directory = Path(token_data_dir)
    if not directory.exists():
        raise FileNotFoundError(
            f"Token data directory does not exist: {directory}")
    payloads: list[Mapping[str, Any]] = []
    source_files: list[str] = []
    for name in ("train_tokens.pt", "eval_tokens.pt"):
        path = directory / name
        if path.exists():
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, Mapping) or "indices" not in loaded:
                raise ValueError(
                    f"Token artifact must contain an 'indices' tensor: {path}")
            payloads.append(loaded)
            source_files.append(str(path))
    if not payloads:
        raise FileNotFoundError(
            f"No train_tokens.pt or eval_tokens.pt artifacts found below {directory}."
        )
    indices = torch.cat(
        [cast(Tensor, payload["indices"]).detach().cpu().long() for payload in payloads], dim=0
    )
    labels = concatenate_optional_tensor(payloads, "labels")
    data = concatenate_optional_tensor(payloads, "data")
    return TokenArtifactData(indices=indices, labels=labels, data=data, source_files=source_files)


def concatenate_optional_tensor(payloads: Sequence[Mapping[str, Any]], key: str) -> Tensor | None:
    """Concatenate a tensor key when it is present in all payloads."""
    if not all(key in payload for payload in payloads):
        return None
    return torch.cat(
        [cast(Tensor, payload[key]).detach().cpu() for payload in payloads],
        dim=0,
    )


def synthetic_token_artifacts(
    *,
    codebook_size: int = 64,
    sequence_length: int = 60,
    batch_size: int = 128,
    quantizer_type: str = "vector",
    num_quantizers: int = 1,
    groups: int = 1,
    seed: int = 0,
) -> tuple[CodebookGeometry, TokenArtifactData, dict[str, Any]]:
    """Create deterministic synthetic codebooks and tokens for public smoke tests."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, codebook_size, endpoint=False)
    base = np.column_stack(
        [
            np.cos(angles),
            np.sin(angles),
            np.cos(2.0 * angles),
            np.sin(2.0 * angles),
        ]
    )
    noise = rng.normal(scale=0.08, size=(codebook_size, 12))
    vector_codebook = np.concatenate([base, noise], axis=1)
    labels = torch.linspace(10.0, 40.0, batch_size).reshape(batch_size, 1)
    if quantizer_type == "residual_vq":
        codebooks = []
        for quantizer in range(num_quantizers):
            scale = 1.0 / float(quantizer + 1)
            codebooks.append(
                vector_codebook * scale +
                rng.normal(scale=0.03, size=vector_codebook.shape)
            )
        geometry = geometry_from_array(
            np.stack(codebooks),
            quantizer_type=quantizer_type,
            source="synthetic_codebook",
            notes=["Synthetic RVQ codebooks used for smoke diagnostics."],
        )
        indices = torch.stack(
            [
                synthetic_indices(batch_size, sequence_length,
                                  codebook_size, offset=quantizer)
                for quantizer in range(num_quantizers)
            ],
            dim=-1,
        )
    elif quantizer_type == "grouped_residual_vq":
        codebooks = []
        for group in range(groups):
            group_books = []
            for quantizer in range(num_quantizers):
                scale = 1.0 / float(group + quantizer + 1)
                group_books.append(
                    vector_codebook * scale +
                    rng.normal(scale=0.03, size=vector_codebook.shape)
                )
            codebooks.append(np.stack(group_books))
        geometry = geometry_from_array(
            np.stack(codebooks),
            quantizer_type=quantizer_type,
            source="synthetic_codebook",
            notes=["Synthetic grouped RVQ codebooks used for smoke diagnostics."],
        )
        group_indices = []
        for group in range(groups):
            quantizer_indices = [
                synthetic_indices(
                    batch_size,
                    sequence_length,
                    codebook_size,
                    offset=group + quantizer,
                )
                for quantizer in range(num_quantizers)
            ]
            group_indices.append(torch.stack(quantizer_indices, dim=-1))
        indices = torch.stack(group_indices, dim=2)
    else:
        geometry = geometry_from_array(
            vector_codebook,
            quantizer_type="vector",
            source="synthetic_codebook",
            notes=["Synthetic vector codebook used for smoke diagnostics."],
        )
        indices = synthetic_indices(
            batch_size, sequence_length, codebook_size, offset=0)
    data = torch.zeros((batch_size, sequence_length, 1), dtype=torch.float32)
    metadata = {
        "quantizer_type": quantizer_type,
        "codebook_size": codebook_size,
        "num_quantizers": num_quantizers,
        "groups": groups,
        "sequence_length": sequence_length,
        "synthetic": True,
    }
    return (
        geometry,
        TokenArtifactData(
            indices=indices.long(),
            labels=labels.float(),
            data=data,
            source_files=["synthetic"],
        ),
        metadata,
    )


def synthetic_indices(
    batch_size: int,
    sequence_length: int,
    codebook_size: int,
    *,
    offset: int,
) -> Tensor:
    """Create smooth deterministic token paths through a circular codebook."""
    time = torch.arange(sequence_length).reshape(1, sequence_length)
    path = torch.arange(batch_size).reshape(batch_size, 1)
    return (time + path * (offset + 1) + offset * 7) % codebook_size


def write_codebook_summary(
    path: str | Path,
    *,
    geometry: CodebookGeometry,
    projection: np.ndarray,
    projection_method: str,
    metadata: Mapping[str, Any],
    usage: Mapping[str, Any],
    condition_buckets: Sequence[Mapping[str, Any]],
    generated_plots: Sequence[str],
    unavailable: Sequence[str],
) -> None:
    """Write a JSON summary of codebook geometry and usage diagnostics."""
    distances = pairwise_distance_summary(geometry.embeddings)
    payload = {
        "metadata": dict(metadata),
        "geometry": {
            "source": geometry.source,
            "embedding_shape": list(geometry.embeddings.shape),
            "entry_count": int(geometry.embeddings.shape[0]),
            "embedding_dim": int(geometry.embeddings.shape[1]),
            "notes": list(geometry.notes),
            "pairwise_distances": distances,
            "projection_method": projection_method,
            "projection_shape": list(projection.shape),
        },
        "usage": dict(usage),
        "condition_buckets": list(condition_buckets),
        "generated_plots": list(generated_plots),
        "unavailable_diagnostics": list(unavailable),
    }
    write_json(path, payload)


def write_projection_csv(
    path: str | Path, geometry: CodebookGeometry, projection: np.ndarray
) -> None:
    """Write 2D codebook projection coordinates."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry", "label", "code",
                        "quantizer", "group", "x", "y"])
        for row, (x_value, y_value) in enumerate(projection):
            writer.writerow(
                [
                    row,
                    geometry.labels[row],
                    geometry.code_indices[row],
                    none_to_empty(geometry.quantizer_indices[row]),
                    none_to_empty(geometry.group_indices[row]),
                    f"{float(x_value):.12g}",
                    f"{float(y_value):.12g}",
                ]
            )


def plot_codebook_projection(
    path: str | Path,
    *,
    geometry: CodebookGeometry,
    projection: np.ndarray,
    title: str = "Codebook PCA projection",
) -> None:
    """Plot projected codebook entries."""
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = component_colours(geometry)
    ax.scatter(
        projection[:, 0], projection[:, 1], c=colours, s=42, edgecolor="white", linewidth=0.4
    )
    for index, label in enumerate(geometry.labels):
        if len(geometry.labels) <= 96:
            ax.text(projection[index, 0], projection[index,
                    1], label, fontsize=6, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_usage_histogram(
    path: str | Path,
    *,
    counts: Sequence[int],
    title: str = "Code usage",
) -> None:
    """Plot marginal token usage counts."""
    apply_clean_style()
    count_array = np.asarray(counts, dtype=np.int64)
    active = np.nonzero(count_array > 0)[0]
    fig, ax = plt.subplots(figsize=(10, 4))
    if active.size == 0:
        ax.text(0.5, 0.5, "No observed codes", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(active, count_array[active], width=1.0, color="#4c78a8")
        ax.set_xlabel("Code index")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} ({active.size} active codes)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_usage_projection(
    path: str | Path,
    *,
    geometry: CodebookGeometry,
    projection: np.ndarray,
    usage: Mapping[str, Any],
) -> None:
    """Overlay usage probabilities on the codebook projection."""
    probabilities = usage_probabilities_for_geometry(geometry, usage)
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    sizes = 28.0 + 900.0 * np.sqrt(probabilities)
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=probabilities,
        s=sizes,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Code usage on PCA projection")
    fig.colorbar(scatter, ax=ax, fraction=0.046,
                 pad=0.04, label="Usage probability")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_vix_bucket_usage(
    path: str | Path,
    *,
    buckets: Sequence[Mapping[str, Any]],
) -> bool:
    """Plot code usage by VIX/condition bucket."""
    if not buckets:
        return False
    matrix = np.asarray([bucket["code_usage_probability"]
                        for bucket in buckets], dtype=np.float64)
    labels = [str(bucket["bucket_label"]) for bucket in buckets]
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.55 * len(labels))))
    image = ax.imshow(matrix, aspect="auto",
                      interpolation="nearest", cmap="magma")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Code index")
    ax.set_ylabel("Condition bucket")
    ax.set_title("Code usage by VIX bucket")
    fig.colorbar(image, ax=ax, fraction=0.046,
                 pad=0.04, label="Usage probability")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def plot_nearest_region(
    path: str | Path,
    *,
    projection: np.ndarray,
    geometry: CodebookGeometry,
) -> None:
    """Plot bounded nearest-code regions in projected space."""
    x_values = projection[:, 0]
    y_values = projection[:, 1]
    x_pad = max((float(x_values.max() - x_values.min()) * 0.08), 1e-3)
    y_pad = max((float(y_values.max() - y_values.min()) * 0.08), 1e-3)
    x_grid = np.linspace(float(x_values.min()) - x_pad,
                         float(x_values.max()) + x_pad, 260)
    y_grid = np.linspace(float(y_values.min()) - y_pad,
                         float(y_values.max()) + y_pad, 260)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    distances = ((grid[:, None, :] - projection[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1).reshape(xx.shape)
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(
        nearest,
        extent=(
            float(x_grid.min()),
            float(x_grid.max()),
            float(y_grid.min()),
            float(y_grid.max()),
        ),
        origin="lower",
        interpolation="nearest",
        cmap="tab20",
        alpha=0.28,
        aspect="auto",
    )
    ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=component_colours(geometry),
        s=34,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Projected nearest-code regions")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_voronoi_or_fallback(
    path: str | Path,
    fallback_path: str | Path,
    *,
    projection: np.ndarray,
    geometry: CodebookGeometry,
) -> str:
    """Plot exact Voronoi regions when available, otherwise nearest-region fallback."""
    if projection.shape[0] < 3:
        plot_nearest_region(
            fallback_path, projection=projection, geometry=geometry)
        return str(Path(fallback_path).name)
    try:
        from scipy.spatial import Voronoi, voronoi_plot_2d

        diagram = Voronoi(projection)
        apply_clean_style()
        fig = voronoi_plot_2d(
            diagram,
            show_vertices=False,
            line_colors="#555555",
            line_width=0.8,
            line_alpha=0.8,
            point_size=20,
        )
        ax = cast(Any, fig).axes[0]
        ax.scatter(projection[:, 0], projection[:, 1],
                   c=component_colours(geometry), s=34)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Projected codebook Voronoi diagram")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return str(Path(path).name)
    except Exception:
        plot_nearest_region(
            fallback_path, projection=projection, geometry=geometry)
        return str(Path(fallback_path).name)


def plot_token_trajectories(
    path: str | Path,
    *,
    indices: Tensor,
    geometry: CodebookGeometry,
    projection: np.ndarray,
    labels: Tensor | None,
    max_paths: int = 8,
) -> bool:
    """Plot example token paths through projected codebook space."""
    if indices.ndim not in {2, 3, 4}:
        return False
    lookup = projection_lookup(geometry, projection)
    token_paths = projected_token_paths(indices, lookup, max_paths=max_paths)
    if not token_paths:
        return False
    condition_values_tensor = condition_values(
        labels) if labels is not None else None
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(projection[:, 0], projection[:, 1], c="#d8d8d8", s=24, zorder=1)
    colour_values = np.linspace(0.0, 1.0, max(len(token_paths), 1))
    colour_map = plt.get_cmap("viridis")
    for path_index, coords in enumerate(token_paths):
        label = f"path {path_index}"
        if condition_values_tensor is not None and path_index < condition_values_tensor.numel():
            label = f"{label}, VIX {float(condition_values_tensor[path_index].item()):.2f}"
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            alpha=0.76,
            color=colour_map(float(colour_values[path_index])),
            label=label,
            zorder=2,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Token trajectory examples")
    if len(token_paths) <= 8:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def plot_q0_q1_pair_heatmap(
    path: str | Path,
    *,
    pair_summary: Mapping[str, Any],
) -> bool:
    """Plot q0/q1 pair counts for two-level RVQ indices."""
    if "pair_counts" not in pair_summary:
        return False
    matrix = np.asarray(pair_summary["pair_counts"], dtype=np.float64)
    apply_clean_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(np.log1p(matrix), aspect="auto",
                      interpolation="nearest", cmap="viridis")
    ax.set_xlabel("q1 code")
    ax.set_ylabel("q0 code")
    ax.set_title("RVQ q0/q1 pair usage (log1p count)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def write_markdown_summary(
    path: str | Path,
    *,
    tokenizer_path: str,
    token_data_path: str,
    metadata: Mapping[str, Any],
    index_shape: Sequence[int],
    usage: Mapping[str, Any],
    generated_plots: Sequence[str],
    unavailable: Sequence[str],
) -> None:
    """Write a compact Markdown summary for the latent-geometry run."""
    lines = [
        "# Discrete Latent Geometry Summary",
        "",
        f"- tokenizer path: `{tokenizer_path}`",
        f"- token data path: `{token_data_path}`",
        f"- quantizer type: `{metadata.get('quantizer_type', 'unknown')}`",
        f"- index shape: `{list(index_shape)}`",
        f"- active codes: `{usage.get('active_code_count', 0)}`",
        f"- perplexity: `{float(usage.get('codebook_perplexity', 0.0)):.8f}`",
        f"- entropy: `{float(usage.get('index_entropy', 0.0)):.8f}`",
        "",
        "## Plots Generated",
        "",
    ]
    lines.extend(f"- `{plot}`" for plot in generated_plots)
    lines.extend(["", "## Unavailable Diagnostics", ""])
    if unavailable:
        lines.extend(f"- {item}" for item in unavailable)
    else:
        lines.append("- none")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def pairwise_distance_summary(embeddings: np.ndarray) -> dict[str, float]:
    """Return compact pairwise Euclidean distance statistics."""
    if embeddings.shape[0] <= 1:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    differences = embeddings[:, None, :] - embeddings[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=2))
    upper = distances[np.triu_indices(embeddings.shape[0], k=1)]
    return {
        "min": float(np.min(upper)),
        "mean": float(np.mean(upper)),
        "max": float(np.max(upper)),
    }


def condition_values(labels: Tensor | None) -> Tensor:
    """Collapse labels to one scalar condition per path."""
    if labels is None:
        return torch.empty(0)
    detached = labels.detach().cpu().float()
    if detached.ndim == 1:
        return detached
    if detached.ndim == 2:
        return detached.reshape(detached.shape[0], -1).mean(dim=1)
    if detached.ndim == 3:
        return detached.mean(dim=(1, 2))
    raise ValueError(
        "Labels must be [batch], [batch, condition_dim], or [batch, length, condition_dim]; "
        f"got {tuple(detached.shape)}."
    )


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with tensor and NumPy conversion."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def json_safe(value: Any) -> Any:
    """Convert tensors, arrays, and paths to JSON-safe values."""
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def to_numpy(value: np.ndarray | Tensor) -> np.ndarray:
    """Convert a tensor-like value to a CPU NumPy array."""
    if isinstance(value, Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float64)


def code_label(code: int, *, quantizer: int | None, group: int | None) -> str:
    """Return a compact label for a codebook entry."""
    if group is not None and quantizer is not None:
        return f"g{group}.q{quantizer}.c{code}"
    if quantizer is not None:
        return f"q{quantizer}.c{code}"
    return str(code)


def none_to_empty(value: int | None) -> str | int:
    """Return an empty string for optional CSV fields."""
    return "" if value is None else value


def component_colours(geometry: CodebookGeometry) -> list[str]:
    """Return stable colours for vector, RVQ, and grouped entries."""
    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
    ]
    colours = []
    for quantizer, group in zip(geometry.quantizer_indices, geometry.group_indices, strict=True):
        index = 0
        if quantizer is not None:
            index += quantizer
        if group is not None:
            index += 3 * group
        colours.append(palette[index % len(palette)])
    return colours


def usage_probabilities_for_geometry(
    geometry: CodebookGeometry,
    usage: Mapping[str, Any],
) -> np.ndarray:
    """Map marginal or component usage probabilities to flattened geometry rows."""
    marginal = np.asarray(
        usage.get("code_usage_probability", []), dtype=np.float64)
    probabilities = np.zeros(len(geometry.code_indices), dtype=np.float64)
    per_quantizer = {
        int(item["quantizer_index"]): np.asarray(item["code_usage_probability"], dtype=np.float64)
        for item in cast(Sequence[Mapping[str, Any]], usage.get("per_quantizer", []))
        if "quantizer_index" in item
    }
    per_group_quantizer = {
        (int(item["group_index"]), int(item["quantizer_index"])): np.asarray(
            item["code_usage_probability"],
            dtype=np.float64,
        )
        for item in cast(Sequence[Mapping[str, Any]], usage.get("per_group_quantizer", []))
        if "group_index" in item and "quantizer_index" in item
    }
    for row, code in enumerate(geometry.code_indices):
        quantizer = geometry.quantizer_indices[row]
        group = geometry.group_indices[row]
        if (
            group is not None
            and quantizer is not None
            and (group, quantizer) in per_group_quantizer
        ):
            source = per_group_quantizer[(group, quantizer)]
        elif quantizer is not None and quantizer in per_quantizer:
            source = per_quantizer[quantizer]
        else:
            source = marginal
        if 0 <= code < len(source):
            probabilities[row] = float(source[code])
    return probabilities


def projection_lookup(
    geometry: CodebookGeometry,
    projection: np.ndarray,
) -> dict[tuple[int | None, int | None, int], np.ndarray]:
    """Build a component-aware projection lookup."""
    lookup: dict[tuple[int | None, int | None, int], np.ndarray] = {}
    for row, code in enumerate(geometry.code_indices):
        key = (geometry.group_indices[row],
               geometry.quantizer_indices[row], code)
        lookup[key] = projection[row]
    return lookup


def projected_token_paths(
    indices: Tensor,
    lookup: Mapping[tuple[int | None, int | None, int], np.ndarray],
    *,
    max_paths: int,
) -> list[np.ndarray]:
    """Map token sequences to 2D projected trajectories."""
    cpu_indices = indices.detach().cpu().long()
    paths: list[np.ndarray] = []
    for batch_index in range(min(max_paths, cpu_indices.shape[0])):
        coords: list[np.ndarray] = []
        if cpu_indices.ndim == 2:
            for code in cpu_indices[batch_index].tolist():
                point = lookup.get((None, None, int(code)))
                if point is not None:
                    coords.append(point)
        elif cpu_indices.ndim == 3:
            for time_index in range(cpu_indices.shape[1]):
                component_points = []
                for quantizer in range(cpu_indices.shape[2]):
                    code = int(cpu_indices[batch_index,
                               time_index, quantizer].item())
                    point = lookup.get((None, quantizer, code))
                    if point is not None:
                        component_points.append(point)
                if component_points:
                    coords.append(np.mean(np.stack(component_points), axis=0))
        elif cpu_indices.ndim == 4:
            for time_index in range(cpu_indices.shape[1]):
                component_points = []
                for group in range(cpu_indices.shape[2]):
                    for quantizer in range(cpu_indices.shape[3]):
                        code = int(
                            cpu_indices[batch_index, time_index, group, quantizer].item())
                        point = lookup.get((group, quantizer, code))
                        if point is not None:
                            component_points.append(point)
                if component_points:
                    coords.append(np.mean(np.stack(component_points), axis=0))
        if coords:
            paths.append(np.stack(coords))
    return paths


def _direct_codebook_from_tokenizer(tokenizer: nn.Module) -> np.ndarray | None:
    """Try guarded direct extraction from known backend state tensors."""
    quantizer = getattr(tokenizer, "quantizer", None)
    backend = getattr(quantizer, "backend", None)
    if backend is None:
        return None
    state = cast(nn.Module, backend).state_dict()
    grouped = _grouped_residual_state_codebook(state)
    if grouped is not None:
        return grouped
    residual = _residual_state_codebook(state)
    if residual is not None:
        return residual
    candidates: list[tuple[str, Tensor]] = []
    for name, value in state.items():
        if not isinstance(value, Tensor) or value.ndim < 2:
            continue
        lower = name.lower()
        if _looks_like_codebook_embedding_name(lower):
            candidates.append((name, value.detach().cpu()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        item[1].ndim, item[1].numel()), reverse=True)
    return normalise_direct_codebook(candidates[0][1])


def _grouped_residual_state_codebook(state: Mapping[str, Tensor]) -> np.ndarray | None:
    """Collect GroupedResidualVQ codebooks from guarded state-dict names."""
    pattern = re.compile(
        r"^rvqs\.(?P<group>\d+)\.layers\.(?P<quantizer>\d+)\..*\.embed$")
    entries: dict[tuple[int, int], np.ndarray] = {}
    for name, value in state.items():
        match = pattern.match(name)
        if match is None or not isinstance(value, Tensor):
            continue
        entries[(int(match.group("group")), int(match.group("quantizer")))] = _squeeze_codebook(
            value
        )
    if not entries:
        return None
    group_count = max(group for group, _quantizer in entries) + 1
    quantizer_count = max(quantizer for _group, quantizer in entries) + 1
    if len(entries) != group_count * quantizer_count:
        return None
    stacked: np.ndarray = np.stack(
        [
            np.stack([entries[(group, quantizer)]
                     for quantizer in range(quantizer_count)])
            for group in range(group_count)
        ]
    )
    return stacked


def _residual_state_codebook(state: Mapping[str, Tensor]) -> np.ndarray | None:
    """Collect ResidualVQ codebooks from guarded state-dict names."""
    pattern = re.compile(r"^layers\.(?P<quantizer>\d+)\..*\.embed$")
    entries: dict[int, np.ndarray] = {}
    for name, value in state.items():
        match = pattern.match(name)
        if match is None or not isinstance(value, Tensor):
            continue
        entries[int(match.group("quantizer"))] = _squeeze_codebook(value)
    if not entries:
        return None
    quantizer_count = max(entries) + 1
    if len(entries) != quantizer_count:
        return None
    stacked: np.ndarray = np.stack(
        [entries[quantizer] for quantizer in range(quantizer_count)])
    return stacked


def _looks_like_codebook_embedding_name(name: str) -> bool:
    """Return whether a guarded state key likely stores direct codebook vectors."""
    return name.endswith(".embed") and not any(
        excluded in name for excluded in ("cluster_size", "initted", "ema", "embed_avg")
    )


def _squeeze_codebook(value: Tensor) -> np.ndarray:
    """Squeeze backend codebook tensors such as ``[1, code, dim]``."""
    array = value.detach().cpu().float().numpy()
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(
            f"Expected a rank-2 codebook after squeezing; got {squeezed.shape}.")
    return squeezed


def normalise_direct_codebook(value: Tensor) -> np.ndarray | None:
    """Normalise common backend codebook tensor layouts."""
    array = value.float().numpy()
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return array
    return None


def _decode_all_codes(
    tokenizer: nn.Module,
    *,
    codebook_size: int,
    quantizer_type: str,
) -> np.ndarray | None:
    """Decode singleton code probes for vector quantizers."""
    if quantizer_type != "vector":
        return None
    quantizer = getattr(tokenizer, "quantizer", None)
    if quantizer is None or not hasattr(quantizer, "decode_indices"):
        return None
    try:
        probe = torch.arange(
            codebook_size, dtype=torch.long).reshape(codebook_size, 1)
        with torch.no_grad():
            decoded = cast(Any, quantizer).decode_indices(probe).detach().cpu()
        if decoded.ndim != 3:
            return None
        return cast(np.ndarray, decoded[:, 0, :].float().numpy())
    except Exception:
        return None


def _observed_vector_geometry(
    quantized: Tensor,
    indices: Tensor,
    codebook_size: int,
    *,
    notes: Sequence[str],
) -> CodebookGeometry:
    """Average decoded vector quantized embeddings by observed code."""
    embedding_dim = quantized.shape[-1]
    sums = torch.zeros((codebook_size, embedding_dim), dtype=torch.float32)
    counts = torch.zeros(codebook_size, dtype=torch.float32)
    flat_quantized = quantized.reshape(-1, embedding_dim).float()
    flat_indices = indices.reshape(-1).long()
    for code in range(codebook_size):
        mask = flat_indices == code
        if bool(mask.any()):
            sums[code] = flat_quantized[mask].mean(dim=0)
            counts[code] = 1.0
    active = counts > 0.0
    code_indices = [int(code) for code in torch.nonzero(
        active, as_tuple=False).flatten()]
    embeddings = sums[active].numpy()
    return CodebookGeometry(
        embeddings=embeddings,
        code_indices=code_indices,
        quantizer_indices=[None] * len(code_indices),
        group_indices=[None] * len(code_indices),
        labels=[str(code) for code in code_indices],
        source="observed_quantized_mean",
        notes=list(notes),
    )


def _observed_rvq_geometry(
    quantized: Tensor,
    indices: Tensor,
    codebook_size: int,
    *,
    num_quantizers: int,
    notes: Sequence[str],
) -> CodebookGeometry:
    """Average final decoded RVQ embeddings by component code."""
    embedding_dim = quantized.shape[-1]
    embeddings: list[np.ndarray] = []
    code_indices: list[int] = []
    quantizer_indices: list[int | None] = []
    group_indices: list[int | None] = []
    flat_quantized = quantized.reshape(-1, embedding_dim).float()
    for quantizer in range(num_quantizers):
        flat_indices = indices[:, :, quantizer].reshape(-1).long()
        for code in range(codebook_size):
            mask = flat_indices == code
            if bool(mask.any()):
                embeddings.append(flat_quantized[mask].mean(dim=0).numpy())
                code_indices.append(code)
                quantizer_indices.append(quantizer)
                group_indices.append(None)
    return CodebookGeometry(
        embeddings=np.stack(embeddings),
        code_indices=code_indices,
        quantizer_indices=quantizer_indices,
        group_indices=group_indices,
        labels=[
            code_label(code, quantizer=quantizer, group=None)
            for code, quantizer in zip(code_indices, quantizer_indices, strict=True)
        ],
        source="observed_quantized_mean",
        notes=list(notes),
    )


def _observed_grouped_geometry(
    quantized: Tensor,
    indices: Tensor,
    codebook_size: int,
    *,
    groups: int,
    num_quantizers: int,
    notes: Sequence[str],
) -> CodebookGeometry:
    """Average final decoded grouped-RVQ embeddings by group/component code."""
    embedding_dim = quantized.shape[-1]
    embeddings: list[np.ndarray] = []
    code_indices: list[int] = []
    quantizer_indices: list[int | None] = []
    group_indices: list[int | None] = []
    flat_quantized = quantized.reshape(-1, embedding_dim).float()
    for group in range(groups):
        for quantizer in range(num_quantizers):
            flat_indices = indices[:, :, group, quantizer].reshape(-1).long()
            for code in range(codebook_size):
                mask = flat_indices == code
                if bool(mask.any()):
                    embeddings.append(flat_quantized[mask].mean(dim=0).numpy())
                    code_indices.append(code)
                    quantizer_indices.append(quantizer)
                    group_indices.append(group)
    return CodebookGeometry(
        embeddings=np.stack(embeddings),
        code_indices=code_indices,
        quantizer_indices=quantizer_indices,
        group_indices=group_indices,
        labels=[
            code_label(code, quantizer=quantizer, group=group)
            for code, quantizer, group in zip(
                code_indices,
                quantizer_indices,
                group_indices,
                strict=True,
            )
        ],
        source="observed_quantized_mean",
        notes=list(notes),
    )
