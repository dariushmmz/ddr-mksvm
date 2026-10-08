# Paper figure index

All figures were generated from frozen machine-readable artifacts by `generate_paper_figures.py`. No model fitting or parameter selection occurs in the figure script. Each figure is available as a 300 dpi PNG for the DOCX/PDF and as an SVG for journal production.

| Figure | Manuscript role | Data source | Files |
|---|---|---|---|
| 1. RKHS-RBF-OVO paired error deltas | Forest plot of seven untouched 96-seed Source-Profile q1 versus RKHS-RBF-OVO confirmations with paired 95% confidence intervals | `results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv` | `figures/fig1_rkhs_error_deltas.png`, `.svg` |
| 2. Class-sensitive tradeoff | Error change against balanced-accuracy change on seven 96-seed unweighted versus class-sensitive RKHS-RBF-OVO confirmations | same consolidated paired-statistics file | `figures/fig2_class_sensitive_tradeoff.png`, `.svg` |
| 3. Runtime ratios | RKHS-RBF-OVO/Source-Profile q1 and class-sensitive/unweighted RKHS mean runtime ratios; logarithmic x-axis | same consolidated paired-statistics file | `figures/fig3_runtime_ratios.png`, `.svg` |
| 4. Bounded-RBF robust Gate8 deltas | Bounded-RBF Class-Sensitive Robust q1 SVM minus its weighted deterministic RBF q1 control across four fresh Gate8 datasets | `results/robust/bounded_rbf_redesign/gates/r9-gate8-15300-15307-v1/paired_statistics.csv` | `figures/fig4_bounded_rbf_gate8_deltas.png`, `.svg` |
| 5. Robust-failure mechanism | Robust/reference ratios for coefficient L1 norm, support count, and slack; corrected versus introduced errors | `results/robust/bounded_rbf_redesign/r10_failure_analysis/paired_objective_deltas.csv`; `dataset_flip_summary.csv` | `figures/fig5_robust_failure_mechanism.png`, `.svg` |

## Rendering notes

- The palette is color-vision-friendly and remains distinguishable in grayscale through position and labeling.
- Figure 1 uses candidate-minus-reference error, so negative values favor the RKHS-RBF-OVO SVM.
- Figure 2 uses class-sensitive-minus-unweighted error on the horizontal axis (lower is better) and balanced-accuracy change on the vertical axis (higher is better).
- Figure 4 uses bounded-RBF robust minus weighted deterministic values; positive error and negative score deltas are adverse.
- Figure 5 reports ratios of seed-averaged quantities and pooled test-observation flip counts. It is diagnostic, not a causal estimate.
- Vector SVGs should be preferred if a target journal accepts them; the embedded PNGs are 300 dpi.
