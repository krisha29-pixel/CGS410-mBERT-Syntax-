# Emergent Dependency Structures in mBERT Attention

**CGS 410 — Computational Linguistics Final Project**

## Research Question

> To what extent do individual attention heads in `bert-base-multilingual-cased`
> recover gold-standard dependency arcs, and does their accuracy vary
> systematically with dependency length, syntactic relation type, and
> language-typological properties?

## Languages

| Language | Treebank | Typology |
|----------|----------|----------|
| English  | UD_English-EWT | SVO, configurational |
| French   | UD_French-GSD  | SVO, configurational |
| Spanish  | UD_Spanish-AnCora | SVO, fusional |
| Hindi    | UD_Hindi-HDTB  | SOV, relatively free order |
| Japanese | UD_Japanese-GSD | SOV, head-final |
| Korean   | UD_Korean-Kaist | SOV, agglutinative |

## Pipeline Overview

```
01_preprocess_treebank.py    →  .jsonl (filtered sentences + gold trees + frag_ratio)
02_extract_attention.py      →  .npz  (word-level attention tensors)
03_induce_trees.py           →  .jsonl (predicted trees per head)
04_evaluate_uas.py           →  .csv/.json (UAS grids + baselines)
05_build_regression_df.py    →  .csv  (arc-level regression data)
06_run_lmem.py               →  .txt/.csv (OLS / mixed-effects model results)
07_generate_figures.py       →  .pdf/.png (publication figures)
08_best_head_per_relation.py →  console (relation-specific head analysis)
09_visualize_tree.py         →  .pdf (qualitative tree comparison)
10_head_transfer.py          →  .csv (cross-lingual head transfer results)
11_visualize_head_clusters.py→  .pdf (PCA "Brain Map" cluster visualization)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place UD treebanks under data/raw/
#    e.g. data/raw/UD_English-EWT/*.conllu

# 3. Run the full pipeline
for lang in en hi fr ja ko es; do
    python src/01_preprocess_treebank.py --input data/raw/UD_${lang}* --output data/processed/${lang}.jsonl --lang ${lang}
    python src/02_extract_attention.py --input data/processed/${lang}.jsonl --output data/attention/${lang}_attention.npz
    python src/03_induce_trees.py --attention data/attention/${lang}_attention.npz --gold data/processed/${lang}.jsonl --output results/predicted_trees/${lang}_trees.jsonl
    python src/04_evaluate_uas.py --predictions results/predicted_trees/${lang}_trees.jsonl --gold data/processed/${lang}.jsonl --output-dir results/uas_tables/ --lang ${lang}
done

python src/05_build_regression_df.py --predictions-dir results/predicted_trees/ --gold-dir data/processed/ --uas-dir results/uas_tables/ --output results/regression/regression_data.csv
python src/06_run_lmem.py --input results/regression/regression_data.csv --output-dir results/regression/
python src/07_generate_figures.py --uas-dir results/uas_tables/ --regression-dir results/regression/ --output-dir results/figures/
python src/10_head_transfer.py --predictions-dir results/predicted_trees/ --gold-dir data/processed/ --uas-dir results/uas_tables/ --output results/transfer/transfer_results.csv
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- GPU recommended for attention extraction (CPU works but is slower)
- R + lme4 for the full GLMM analysis (optional — statsmodels fallback included)

## Project Structure

```
project/
├── configs/experiment_config.yaml
├── data/
│   ├── raw/           # UD .conllu files (not tracked — download separately)
│   ├── processed/     # Filtered .jsonl
│   └── attention/     # Attention .npz
├── src/
│   ├── 01–11 pipeline scripts
│   └── utils/
│       ├── mst.py             # Chu-Liu/Edmonds algorithm
│       ├── attention_utils.py # Subword-to-word aggregation
│       ├── conllu_utils.py    # CoNLL-U parsing & filtering
│       └── eval_utils.py      # UAS scoring & baselines
├── report/
│   └── CGS410_Final_Report.tex
├── results/
│   └── figures/       # Publication-quality PDF/PNG figures
├── requirements.txt
├── run_pipeline.ps1
└── README.md
```

## Authors

- **Anusha Gupta** (240158 )
- **Dhruv Paliwal** (240358)
- **Krisha Gothi** (240562)
- **Manisha Rathod** (240625)

*CGS 410 — Computational Linguistics Final Project*
