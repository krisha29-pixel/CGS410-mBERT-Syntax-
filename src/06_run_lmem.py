#!/usr/bin/env python3
"""
06_run_lmem.py
==============
Fit the mixed-effects models for H₁ (arc-level GLMM) and H₂
(sentence-level LMM) and export results.

Usage
-----
    python src/06_run_lmem.py \
        --input      results/regression/regression_data.csv \
        --output-dir results/regression/ \
        --config     configs/experiment_config.yaml

Requirements
------------
- R must be installed and on PATH.
- R packages: lme4, lmerTest
- Python: pymer4 (pip install pymer4)

If pymer4 is not available, the script falls back to a simpler
statsmodels OLS + sentence-level LMM as a reduced alternative.
"""

import argparse
import yaml
import sys
import os
import math
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def fit_with_pymer4(df, output_dir):
    """Fit models using pymer4 (requires R + lme4)."""
    from pymer4.models import Lmer

    out = Path(output_dir)

    # ── H₁: Arc-level GLMM ─────────────────────────────────
    print("\n=== H1: Logistic GLMM (arc-level) ===")
    formula_h1 = (
        "arc_correct ~ log_dep_length + log_sent_length + language + "
        "deprel_class + (1|sent_id) + (1|treebank)"
    )

    model_h1 = Lmer(formula_h1, data=df, family="binomial")
    model_h1.fit(
        factors={"language": list(df["language"].unique()),
                 "deprel_class": list(df["deprel_class"].unique())},
        summarize=True,
    )

    h1_summary = str(model_h1.summary())
    h1_path = out / "h1_glmm_summary.txt"
    with open(h1_path, "w") as f:
        f.write(h1_summary)
    print(f"  Saved -> {h1_path}")

    # Save coefficients
    coefs = model_h1.coefs
    coefs.to_csv(out / "h1_coefficients.csv")

    # ── H₂: Sentence-level LMM ─────────────────────────────
    print("\n=== H2: Linear MM (sentence-level UAS) ===")
    # Aggregate to sentence level
    sent_df = (
        df.groupby(["sent_id", "language", "treebank", "log_sent_length"])
        .agg(uas=("arc_correct", "mean"), n_arcs=("arc_correct", "count"))
        .reset_index()
    )

    formula_h2 = "uas ~ language + log_sent_length + (1|treebank)"
    model_h2 = Lmer(formula_h2, data=sent_df)
    model_h2.fit(
        factors={"language": list(sent_df["language"].unique())},
        summarize=True,
    )

    h2_summary = str(model_h2.summary())
    h2_path = out / "h2_lmm_summary.txt"
    with open(h2_path, "w") as f:
        f.write(h2_summary)
    print(f"  Saved -> {h2_path}")

    model_h2.coefs.to_csv(out / "h2_coefficients.csv")


def fit_with_statsmodels(df, output_dir):
    """Fallback: sentence-level LMM using statsmodels (no R needed)."""
    import statsmodels.formula.api as smf

    out = Path(output_dir)

    print("\n[!] pymer4 not available – using statsmodels (sentence-level only)")

    # Aggregate to sentence level
    sent_df = (
        df.groupby(["sent_id", "language", "treebank",
                     "log_sent_length"])
        .agg(uas=("arc_correct", "mean"), n_arcs=("arc_correct", "count"))
        .reset_index()
    )

    # ── H₁ approximation: OLS at sentence level ────────────
    print("\n=== H1 (approx): OLS on sentence-level UAS ===")
    # Add mean log_dep_length per sentence
    dep_len_means = (
        df.groupby("sent_id")["log_dep_length"].mean().reset_index()
    )
    dep_len_means.columns = ["sent_id", "mean_log_dep_length"]
    sent_df = sent_df.merge(dep_len_means, on="sent_id")

    ols_h1 = smf.ols(
        "uas ~ mean_log_dep_length + log_sent_length + C(language)",
        data=sent_df,
    ).fit()

    h1_path = out / "h1_ols_summary.txt"
    with open(h1_path, "w") as f:
        f.write(ols_h1.summary().as_text())
    print(f"  Saved -> {h1_path}")

    # ── H₂: Mixed LMM ──────────────────────────────────────
    print("\n=== H2: statsmodels MixedLM ===")
    lmm_h2 = smf.mixedlm(
        "uas ~ C(language) + log_sent_length",
        data=sent_df,
        groups=sent_df["treebank"],
    ).fit()

    h2_path = out / "h2_lmm_summary.txt"
    with open(h2_path, "w") as f:
        f.write(lmm_h2.summary().as_text())
    print(f"  Saved -> {h2_path}")

    # ── Extended models with fragmentation covariate ────────
    has_frag = "frag_ratio" in sent_df.columns
    if not has_frag and "frag_ratio" in df.columns:
        frag_means = (
            df.groupby("sent_id")["frag_ratio"].mean().reset_index()
        )
        sent_df = sent_df.merge(frag_means, on="sent_id", how="left")
        has_frag = "frag_ratio" in sent_df.columns

    if has_frag:
        print("\n=== H1 (extended): OLS + frag_ratio ===")
        ols_h1_ext = smf.ols(
            "uas ~ mean_log_dep_length + log_sent_length + C(language) + frag_ratio",
            data=sent_df,
        ).fit()

        h1_ext_path = out / "h1_ols_extended_summary.txt"
        with open(h1_ext_path, "w") as f:
            f.write(ols_h1_ext.summary().as_text())
        print(f"  Saved -> {h1_ext_path}")

        print("\n=== H2 (extended): MixedLM + frag_ratio ===")
        lmm_h2_ext = smf.mixedlm(
            "uas ~ C(language) + log_sent_length + frag_ratio",
            data=sent_df,
            groups=sent_df["treebank"],
        ).fit()

        h2_ext_path = out / "h2_lmm_extended_summary.txt"
        with open(h2_ext_path, "w") as f:
            f.write(lmm_h2_ext.summary().as_text())
        print(f"  Saved -> {h2_ext_path}")
    else:
        print("\n[!] frag_ratio not found in data — skipping extended models")


def main():
    parser = argparse.ArgumentParser(
        description="Fit mixed-effects models for H₁ and H₂."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config", default="configs/experiment_config.yaml",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input} …")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} arcs across {df['language'].nunique()} languages")

    try:
        fit_with_pymer4(df, args.output_dir)
    except ImportError:
        fit_with_statsmodels(df, args.output_dir)

    print("\n[V] Statistical analysis complete.")


if __name__ == "__main__":
    main()
