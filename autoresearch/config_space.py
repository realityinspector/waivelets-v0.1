"""
Configuration space for waivelets autoresearch.

Defines all mutable analysis parameters: wavelet choice, scale range,
correlation threshold, embedding model, fractal estimation method, etc.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass
class Param:
    key: str
    type: str  # "float", "int", "choice"
    low: float = 0.0
    high: float = 1.0
    choices: list = field(default_factory=list)
    cluster: str = ""
    description: str = ""

    def sample(self, rng: random.Random):
        if self.type == "float":
            return rng.uniform(self.low, self.high)
        elif self.type == "int":
            return rng.randint(int(self.low), int(self.high))
        elif self.type == "choice":
            return rng.choice(self.choices)
        raise ValueError(f"Unknown param type: {self.type}")


# ── Wavelet transform parameters ──
WAVELET_PARAMS = [
    Param("wavelet", "choice",
          choices=["morl", "cmor1.5-1.0", "cmor1.5-0.5", "gaus1", "gaus2",
                   "gaus4", "gaus8", "mexh", "shan1-1.5", "fbsp1-1-1.5"],
          cluster="wavelet", description="Mother wavelet function"),
    Param("max_scale_ratio", "float", low=0.1, high=1.0,
          cluster="wavelet",
          description="Max scale as fraction of n_units/2 (0.5 = quarter of text)"),
    Param("min_scale", "int", low=1, high=4,
          cluster="wavelet", description="Minimum wavelet scale"),
    Param("scale_step", "choice", choices=["linear", "log", "dyadic"],
          cluster="wavelet",
          description="How to space scales — linear, log, or dyadic (powers of 2)"),
]

# ── Graph construction parameters ──
GRAPH_PARAMS = [
    Param("correlation_threshold", "float", low=0.2, high=0.8,
          cluster="graph",
          description="Threshold for dimension correlation → adjacency"),
    Param("correlation_method", "choice",
          choices=["pearson", "spearman", "cosine"],
          cluster="graph", description="Correlation method for dimension-dimension similarity"),
    Param("weight_edges", "choice", choices=[True, False],
          cluster="graph", description="Use weighted vs binary adjacency"),
]

# ── Fractal estimation parameters ──
FRACTAL_PARAMS = [
    Param("fractal_fit_range_low", "float", low=0.0, high=0.3,
          cluster="fractal",
          description="Start of scale range for fractal fit (fraction of total)"),
    Param("fractal_fit_range_high", "float", low=0.5, high=1.0,
          cluster="fractal",
          description="End of scale range for fractal fit (fraction of total)"),
    Param("fractal_method", "choice",
          choices=["polyfit", "robust_linregress", "detrended"],
          cluster="fractal",
          description="Method for estimating scaling exponent"),
]

# ── Baseline / significance parameters ──
BASELINE_PARAMS = [
    Param("baseline_trials", "int", low=3, high=8,
          cluster="baseline", description="Number of shuffled baseline trials"),
    Param("shuffle_method", "choice",
          choices=["permute_rows", "permute_within_dim", "block_shuffle",
                   "phase_randomize"],
          cluster="baseline",
          description="How to destroy sequential structure in null hypothesis"),
    Param("block_size", "int", low=2, high=10,
          cluster="baseline",
          description="Block size for block_shuffle method"),
]

ALL_CLUSTERS = {
    "wavelet": WAVELET_PARAMS,
    "graph": GRAPH_PARAMS,
    "fractal": FRACTAL_PARAMS,
    "baseline": BASELINE_PARAMS,
}


class ConfigSpace:
    """Sample configurations from the parameter space."""

    def __init__(self, clusters: list[str] | None = None, seed: int = 42):
        self.rng = random.Random(seed)
        if clusters:
            self.params = [p for c in clusters for p in ALL_CLUSTERS.get(c, [])]
        else:
            self.params = [p for ps in ALL_CLUSTERS.values() for p in ps]

    def sample(self, n_params: int | None = None) -> dict:
        """Sample a config, mutating n_params randomly (default: 1-3)."""
        if n_params is None:
            n_params = self.rng.randint(1, min(3, len(self.params)))
        chosen = self.rng.sample(self.params, min(n_params, len(self.params)))
        return {p.key: p.sample(self.rng) for p in chosen}

    def sample_full(self) -> dict:
        """Sample all parameters."""
        return {p.key: p.sample(self.rng) for p in self.params}
