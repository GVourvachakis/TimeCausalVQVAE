"""Run S&P500/VIX causal VQ-family tokenizer ablations."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_pdv_tokenizer_ablation.py")),
        run_name="__main__",
    )
