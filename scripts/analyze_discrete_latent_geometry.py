"""Analyse discrete latent geometry for trained or synthetic VQ-family tokenizers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from time_causal_vae.evaluation.latent_geometry import (
    CodebookGeometry,
    TokenArtifactData,
    code_usage_summary,
    condition_bucket_usage,
    extract_codebook_geometry,
    plot_codebook_projection,
    plot_nearest_region,
    plot_q0_q1_pair_heatmap,
    plot_token_trajectories,
    plot_usage_histogram,
    plot_usage_projection,
    plot_vix_bucket_usage,
    plot_voronoi_or_fallback,
    project_embeddings,
    q0_q1_pair_summary,
    synthetic_token_artifacts,
    write_codebook_summary,
    write_json,
    write_markdown_summary,
    write_projection_csv,
)

DEFAULT_WANDB_PROJECT = "time-causal-latent-diagnostics"
DEFAULT_WANDB_ENTITY = "tc_vae"


def build_parser() -> argparse.ArgumentParser:
    """Build the latent-geometry CLI parser."""
    parser = argparse.ArgumentParser(
        description="Analyse geometry, usage, and trajectories in discrete tokenizer codebooks.",
    )
    parser.add_argument("--tokenizer-dir", help="Directory containing tokenizer.pt.")
    parser.add_argument("--token-data-dir", help="Directory containing extracted token artifacts.")
    parser.add_argument("--config", help="Tokenizer experiment YAML config.")
    parser.add_argument("--output-dir", required=True, help="Output directory under outputs/.")
    parser.add_argument(
        "--base-data-dir",
        default="data/processed",
        help="Base data directory recorded for reproducibility; no training is performed.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run deterministic synthetic smoke diagnostics without local checkpoints.",
    )
    parser.add_argument(
        "--projection",
        choices=["pca"],
        default="pca",
        help="2D projection method.",
    )
    parser.add_argument(
        "--plot-voronoi",
        action="store_true",
        help="Plot exact Voronoi regions when available, otherwise bounded nearest regions.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log summaries and plots to W&B.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--run-name", help="Optional W&B run name.")
    return parser


def main() -> None:
    """Run latent-geometry diagnostics."""
    args = build_parser().parse_args()
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_optional_yaml(args.config)
    if args.synthetic:
        geometry, artifacts, metadata = build_synthetic_inputs(raw_config)
        tokenizer_path = "synthetic"
        token_data_path = "synthetic"
    else:
        require_real_args(args)
        geometry, artifacts, metadata = build_real_inputs(
            tokenizer_dir=cast(str, args.tokenizer_dir),
            token_data_dir=cast(str, args.token_data_dir),
        )
        tokenizer_path = cast(str, args.tokenizer_dir)
        token_data_path = cast(str, args.token_data_dir)
        metadata["config_path"] = str(args.config)
        metadata["config"] = raw_config

    metadata["base_data_dir"] = str(args.base_data_dir)
    generated_plots: list[str] = []
    unavailable: list[str] = []
    usage = code_usage_summary(artifacts.indices, int(metadata["codebook_size"]))
    buckets = condition_bucket_usage(
        indices=artifacts.indices,
        labels=artifacts.labels,
        codebook_size=int(metadata["codebook_size"]),
    )
    if not buckets:
        unavailable.append("usage by VIX bucket unavailable because labels were absent.")

    projection, projection_method = project_embeddings(geometry.embeddings, method=args.projection)
    write_projection_csv(output_dir / "codebook_projection.csv", geometry, projection)
    plot_codebook_projection(
        output_dir / "codebook_projection.png", geometry=geometry, projection=projection
    )
    generated_plots.append("codebook_projection.png")

    plot_usage_histogram(
        output_dir / "code_usage_histogram.png",
        counts=cast(Sequence[int], usage["code_usage_counts"]),
    )
    generated_plots.append("code_usage_histogram.png")
    plot_usage_projection(
        output_dir / "codebook_usage_projection.png",
        geometry=geometry,
        projection=projection,
        usage=usage,
    )
    generated_plots.append("codebook_usage_projection.png")

    if plot_vix_bucket_usage(output_dir / "vix_bucket_code_usage.png", buckets=buckets):
        generated_plots.append("vix_bucket_code_usage.png")

    if plot_token_trajectories(
        output_dir / "token_trajectory_examples.png",
        indices=artifacts.indices,
        geometry=geometry,
        projection=projection,
        labels=artifacts.labels,
    ):
        generated_plots.append("token_trajectory_examples.png")
    else:
        unavailable.append("token trajectories unavailable for the observed index layout.")

    if args.plot_voronoi:
        voronoi_name = plot_voronoi_or_fallback(
            output_dir / "codebook_voronoi.png",
            output_dir / "codebook_nearest_region.png",
            projection=projection,
            geometry=geometry,
        )
        generated_plots.append(voronoi_name)
    else:
        plot_nearest_region(
            output_dir / "codebook_nearest_region.png",
            projection=projection,
            geometry=geometry,
        )
        generated_plots.append("codebook_nearest_region.png")

    if artifacts.indices.ndim == 3 and artifacts.indices.shape[-1] == 2:
        pair_summary = q0_q1_pair_summary(artifacts.indices, int(metadata["codebook_size"]))
        write_json(output_dir / "q0_q1_pair_summary.json", pair_summary)
        if plot_q0_q1_pair_heatmap(
            output_dir / "q0_q1_pair_heatmap.png",
            pair_summary=pair_summary,
        ):
            generated_plots.append("q0_q1_pair_heatmap.png")
    else:
        unavailable.append("RVQ q0/q1 pair analysis requires indices with shape [batch, time, 2].")

    write_codebook_summary(
        output_dir / "codebook_geometry_summary.json",
        geometry=geometry,
        projection=projection,
        projection_method=projection_method,
        metadata=metadata,
        usage=usage,
        condition_buckets=buckets,
        generated_plots=generated_plots,
        unavailable=unavailable,
    )
    write_markdown_summary(
        output_dir / "latent_geometry_summary.md",
        tokenizer_path=tokenizer_path,
        token_data_path=token_data_path,
        metadata=metadata,
        index_shape=list(artifacts.indices.shape),
        usage=usage,
        generated_plots=generated_plots,
        unavailable=unavailable,
    )

    if args.wandb:
        log_to_wandb(
            output_dir=output_dir,
            run_name=args.run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            metadata=metadata,
            usage=usage,
            generated_plots=generated_plots,
        )

    print("Discrete latent geometry analysis complete.")
    print(f"output_dir: {output_dir}")
    print(f"quantizer_type: {metadata.get('quantizer_type')}")
    print(f"indices_shape: {list(artifacts.indices.shape)}")
    print(
        "code usage: "
        f"active={usage['active_code_count']}/{usage['codebook_size']} "
        f"perplexity={float(usage['codebook_perplexity']):.8f} "
        f"entropy={float(usage['index_entropy']):.8f}"
    )
    print(f"plots: {', '.join(generated_plots)}")


def build_synthetic_inputs(
    raw_config: Mapping[str, Any],
) -> tuple[CodebookGeometry, TokenArtifactData, dict[str, Any]]:
    """Build deterministic synthetic geometry inputs from optional config metadata."""
    model = raw_config.get("model") if isinstance(raw_config.get("model"), Mapping) else {}
    model_mapping = cast(Mapping[str, Any], model)
    quantizer_type = str(model_mapping.get("quantizer_type", "vector"))
    codebook_size = int(model_mapping.get("codebook_size", 64))
    sequence_length = int(model_mapping.get("data_length", 60))
    num_quantizers = int(
        model_mapping.get("num_quantizers", 1 if quantizer_type == "vector" else 2)
    )
    groups = int(model_mapping.get("groups", 1))
    geometry, artifacts, metadata = synthetic_token_artifacts(
        codebook_size=codebook_size,
        sequence_length=sequence_length,
        quantizer_type=quantizer_type,
        num_quantizers=num_quantizers,
        groups=groups,
    )
    metadata["config"] = raw_config
    return geometry, artifacts, metadata


def build_real_inputs(
    *,
    tokenizer_dir: str,
    token_data_dir: str,
) -> tuple[CodebookGeometry, TokenArtifactData, dict[str, Any]]:
    """Load a trained tokenizer and extracted token artifacts."""
    from time_causal_vae.evaluation.tokenizer import load_trained_tokenizer

    artifacts = load_real_token_artifacts(token_data_dir)
    device = torch.device("cpu")
    tokenizer, tokenizer_config, _checkpoint = load_trained_tokenizer(tokenizer_dir, device=device)
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    metadata = config_to_metadata(tokenizer_config)
    geometry = extract_codebook_geometry(
        tokenizer,
        observed_indices=artifacts.indices,
        codebook_size=int(metadata["codebook_size"]),
        quantizer_type=str(metadata["quantizer_type"]),
        num_quantizers=int(metadata.get("num_quantizers", 1)),
        groups=int(metadata.get("groups", 1)),
    )
    metadata["tokenizer_dir"] = tokenizer_dir
    metadata["token_data_dir"] = token_data_dir
    metadata["token_source_files"] = artifacts.source_files
    return geometry, artifacts, metadata


def load_real_token_artifacts(token_data_dir: str) -> TokenArtifactData:
    """Load real token artifacts through the evaluation helper."""
    from time_causal_vae.evaluation.latent_geometry import load_token_artifacts

    return load_token_artifacts(token_data_dir)


def config_to_metadata(config: Any) -> dict[str, Any]:
    """Convert tokenizer config dataclass-like objects to JSON metadata."""
    if is_dataclass(config):
        data = asdict(config)
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        data = {
            name: getattr(config, name)
            for name in (
                "quantizer_type",
                "codebook_size",
                "num_quantizers",
                "groups",
                "data_length",
                "embedding_dim",
            )
            if hasattr(config, name)
        }
    return cast(dict[str, Any], data)


def load_optional_yaml(path: str | None) -> dict[str, Any]:
    """Load optional YAML config metadata."""
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, Mapping):
        raise SystemExit(f"--config must point to a YAML mapping: {path}")
    return dict(loaded)


def require_real_args(args: argparse.Namespace) -> None:
    """Validate required real-analysis arguments."""
    missing = [
        name
        for name in ("tokenizer_dir", "token_data_dir", "config")
        if getattr(args, name) is None
    ]
    if missing:
        formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise SystemExit(
            f"Real tokenizer analysis requires {formatted}; use --synthetic for smoke runs."
        )


def validate_output_dir(output_dir: str) -> Path:
    """Validate that artifacts stay below ignored outputs/."""
    path = Path(output_dir)
    resolved = path.resolve()
    outputs_root = (Path.cwd() / "outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError as exc:
        raise SystemExit(
            f"--output-dir must be under ignored outputs/. Received: {output_dir}"
        ) from exc
    return path


def log_to_wandb(
    *,
    output_dir: Path,
    run_name: str | None,
    project: str,
    entity: str,
    metadata: Mapping[str, Any],
    usage: Mapping[str, Any],
    generated_plots: Sequence[str],
) -> None:
    """Log latent-geometry summaries and generated plots to W&B."""
    try:
        import wandb
    except Exception as exc:
        raise SystemExit(
            "W&B logging requested, but wandb is not importable. "
            "Install the tracking dependency group or omit --wandb."
        ) from exc
    run = wandb.init(project=project, entity=entity, name=run_name, config=dict(metadata))
    wandb.log({
        "active_code_count": usage["active_code_count"],
        "codebook_perplexity": usage["codebook_perplexity"],
        "index_entropy": usage["index_entropy"],
    })
    image_payload = {
        Path(plot_name).stem: wandb.Image(str(output_dir / plot_name))
        for plot_name in generated_plots
        if (output_dir / plot_name).exists()
    }
    if image_payload:
        wandb.log(image_payload)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
