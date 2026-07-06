# CGS 410 Full Pipeline Script (Windows)
# Run this from your project root: .\run_pipeline.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 0: Preprocess Treebanks            " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# python src/01_preprocess_treebank.py --input data/raw/UD_English-EWT --output data/processed/en.jsonl --lang en
# python src/01_preprocess_treebank.py --input data/raw/UD_French-GSD --output data/processed/fr.jsonl --lang fr
# python src/01_preprocess_treebank.py --input data/raw/UD_Hindi-HDTB --output data/processed/hi.jsonl --lang hi
# python src/01_preprocess_treebank.py --input data/raw/UD_Japanese-GSD --output data/processed/ja.jsonl --lang ja
# python src/01_preprocess_treebank.py --input data/raw/UD_Korean-Kaist --output data/processed/ko.jsonl --lang ko
# python src/01_preprocess_treebank.py --input data/raw/UD_Spanish-AnCora --output data/processed/es.jsonl --lang es


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 1: Extracting mBERT Attention      " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "(This is the slowest step - uses GPU if available)"

# python src/02_extract_attention.py --input data/processed/en.jsonl --output data/attention/en_attention.npz
# python src/02_extract_attention.py --input data/processed/fr.jsonl --output data/attention/fr_attention.npz
# python src/02_extract_attention.py --input data/processed/hi.jsonl --output data/attention/hi_attention.npz
# python src/02_extract_attention.py --input data/processed/ja.jsonl --output data/attention/ja_attention.npz
# python src/02_extract_attention.py --input data/processed/ko.jsonl --output data/attention/ko_attention.npz
# python src/02_extract_attention.py --input data/processed/es.jsonl --output data/attention/es_attention.npz


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 2: Induce Dependency Trees         " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# python src/03_induce_trees.py --attention data/attention/en_attention.npz --gold data/processed/en.jsonl --output results/predicted_trees/en_trees.jsonl
# python src/03_induce_trees.py --attention data/attention/fr_attention.npz --gold data/processed/fr.jsonl --output results/predicted_trees/fr_trees.jsonl
# python src/03_induce_trees.py --attention data/attention/hi_attention.npz --gold data/processed/hi.jsonl --output results/predicted_trees/hi_trees.jsonl
# python src/03_induce_trees.py --attention data/attention/ja_attention.npz --gold data/processed/ja.jsonl --output results/predicted_trees/ja_trees.jsonl
# python src/03_induce_trees.py --attention data/attention/ko_attention.npz --gold data/processed/ko.jsonl --output results/predicted_trees/ko_trees.jsonl
# python src/03_induce_trees.py --attention data/attention/es_attention.npz --gold data/processed/es.jsonl --output results/predicted_trees/es_trees.jsonl


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 3: Evaluate UAS Scores             " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# python src/04_evaluate_uas.py --predictions results/predicted_trees/en_trees.jsonl --gold data/processed/en.jsonl --output-dir results/uas_tables/ --lang en
# python src/04_evaluate_uas.py --predictions results/predicted_trees/fr_trees.jsonl --gold data/processed/fr.jsonl --output-dir results/uas_tables/ --lang fr
# python src/04_evaluate_uas.py --predictions results/predicted_trees/hi_trees.jsonl --gold data/processed/hi.jsonl --output-dir results/uas_tables/ --lang hi
# python src/04_evaluate_uas.py --predictions results/predicted_trees/ja_trees.jsonl --gold data/processed/ja.jsonl --output-dir results/uas_tables/ --lang ja
# python src/04_evaluate_uas.py --predictions results/predicted_trees/ko_trees.jsonl --gold data/processed/ko.jsonl --output-dir results/uas_tables/ --lang ko
# python src/04_evaluate_uas.py --predictions results/predicted_trees/es_trees.jsonl --gold data/processed/es.jsonl --output-dir results/uas_tables/ --lang es


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 4: Build Regression DataFrame      " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# python src/05_build_regression_df.py --predictions-dir results/predicted_trees/ --gold-dir data/processed/ --uas-dir results/uas_tables/ --output results/regression/regression_data.csv


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 5: Run Mixed-Effects Model         " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python src/06_run_lmem.py --input results/regression/regression_data.csv --output-dir results/regression/


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 6: Generate Publication Figures    " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python src/07_generate_figures.py --uas-dir results/uas_tables/ --regression-dir results/regression/ --output-dir results/figures/


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 7: Head Transfer Experiment        " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python src/10_head_transfer.py --predictions-dir results/predicted_trees/ --gold-dir data/processed/ --uas-dir results/uas_tables/ --output results/transfer/transfer_results.csv


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 8: Best Head Per Relation          " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python src/08_best_head_per_relation.py


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 9: Head Transfer Experiment        " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python src/10_head_transfer.py


Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Step 10: Head Cluster Visualization     " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

python src/11_visualize_head_clusters.py


Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Pipeline Complete! Check results/figures" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
