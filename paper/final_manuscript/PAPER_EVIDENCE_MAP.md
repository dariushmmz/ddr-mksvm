# Manuscript evidence map

This map connects the article's substantive claims to their authoritative repository evidence. Final/frozen reports and machine-readable results take precedence over intermediate notes. Paths are relative to the project root.

| Claim or manuscript element | Exact evidence | Authority / interpretation |
|---|---|---|
| Nine usable local classification datasets; one metadata CSV rejected | `docs/expanded_dataset_study/DATASET_INVENTORY.md`; `results/expanded_dataset_study/inventory/dataset_inventory.csv`; `dataset_hashes.csv` | final re-audited inventory |
| Final stage disposition for every dataset/comparison | `results/expanded_dataset_study/analysis/cross_dataset/stage_registry.csv`; `docs/expanded_dataset_study/CROSS_DATASET_RESULTS.md` | final protocol accounting |
| RKHS-RBF-OVO seven-dataset 96-seed error, CI, W/T/L, Wilcoxon, effect, runtime | `results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv` | recomputed from frozen per-seed artifacts; primary numerical authority |
| Source-Profile q1 and RKHS-RBF-OVO model means, confusion matrices, class recalls | `results/expanded_dataset_study/analysis/cross_dataset/confirmed_model_summary.csv`; `confirmed_paired_class_recall.csv` | consolidated final analysis |
| Parkinson error 14.243% to 7.079%, delta -7.164 pp | `confirmed_paired_statistics.csv`, dataset `parkinson`, metric `test_error`, comparison `legacy`/`v7` | untouched 96-seed confirmation |
| Breast Cancer recurrence error gain and minority-recall loss | same file plus `confirmed_paired_class_recall.csv`; `docs/expanded_dataset_study/STATISTICAL_ANALYSIS.md` | untouched expanded 96-seed confirmation |
| Dermatology small error gain and class-6 recall loss | same files; `docs/expanded_dataset_study/CROSS_DATASET_RESULTS.md` | untouched expanded 96-seed confirmation; earlier independent replication retained |
| Blood and Mammographic negative RKHS/source-profile Gate8 outcomes | `results/expanded_dataset_study/analysis/cross_dataset/stage_registry.csv`; `docs/V7_CROSS_DATASET_REPORT.md` | development/gating evidence, not confirmation |
| RKHS-RBF-OVO mathematical architecture and frozen selection rules | `docs/V7_MATHEMATICAL_DESIGN.md`; `ddr_mksvm/v7_cross_dataset.py` | frozen design/code; internal file name retained |
| CS-RKHS-RBF-OVO weight formula, tau=0.5, balanced selection | `docs/V8_MATHEMATICAL_DESIGN.md`; `ddr_mksvm/v8_class_sensitive.py` | frozen design/code; internal file name retained |
| CS-RKHS-RBF-OVO seven-dataset paired metrics and runtime | `results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv` | consolidated untouched 96-seed results |
| Blood +5.820 pp balanced accuracy and +17.244 pp minority recall | `confirmed_paired_statistics.csv`; `confirmed_paired_class_recall.csv` | primary class-sensitive confirmation |
| Heart +2.553 pp balanced accuracy, +2.879 pp macro-F1, +1.875 pp error, 4.986x runtime | `confirmed_paired_statistics.csv`; class-recall table | primary class-sensitive confirmation under local five-class protocol |
| Training-Only Dual-Kernel Selection SVM rejected at Gate8 | `docs/V8_1_CROSS_DATASET_REPORT.md`; `docs/V8_1_MATHEMATICAL_DESIGN.md` | prospective negative architecture study; immutable paths retain internal labels |
| Original robust q1 LP, p/rho semantics, active kernels, OVA, threshold and solver discrepancies | `docs/ROBUST_MATLAB_SOURCE_AUDIT.md`; `docs/ROBUST_MATLAB_MATHEMATICAL_AUDIT.md`; `results/robust/source_audit/matlab_source_inventory.json` | authoritative executable-source audit |
| RKHS-Norm Robust Class-Sensitive SVM 24-seed gate stopped; Parkinson 21.14x runtime and 18/24 outer `optimal_inaccurate`; Blood failures | `docs/ROBUST_V8_C_GATE24_REPORT.md`; `docs/robust/01_baseline_and_gate24/` | development gate, not confirmation |
| Independent conic-solver qualification and failure | `docs/robust/02_solver_qualification/SOLVER_QUALIFICATION_RESULTS.md`; `SOLVER_COMPARISON.md`; qualification artifacts under `results/robust/solver_qualification/` | solver-development evidence only |
| High-precision Blood matrices SPD but condition estimates above 1e23 | `docs/robust/02_solver_qualification/HIGH_PRECISION_KERNEL_QUALIFICATION.md`; `results/robust/solver_qualification/r3_3_high_precision/` | high-precision numerical qualification |
| Source-Aligned Class-Sensitive Robust q1 LP polynomial coefficient range and failure | `docs/robust/06_source_aligned_redesign/SMOKE_REPORT.md`; machine results under `results/robust/source_aligned_redesign/` | exposed-seed qualification |
| Exact equilibration passed Parkinson but only 5/8 Blood variants | `docs/robust/06_source_aligned_redesign/LP_EQUILIBRATION_QUALIFICATION.md`; `results/robust/source_aligned_redesign/lp_equilibration/` | final negative equilibration result |
| Bounded-RBF Class-Sensitive Robust q1 SVM construction and reductions | `docs/robust/07_bounded_rbf_redesign/ROBUST_RBF_Q1_DESIGN.md` | frozen pre-gate design |
| Bounded-RBF Gate8 accounting, numerical diagnostics, model means, deltas, runtime | `docs/robust/07_bounded_rbf_redesign/GATE8_REPORT.md`; `results/robust/bounded_rbf_redesign/gates/r9-gate8-15300-15307-v1/` | fresh Gate8 screening evidence |
| Weighted robust candidate completed 32/32, but two Blood deterministic controls failed | `docs/robust/07_bounded_rbf_redesign/GATE8_REPORT.md`; `analysis/task_accounting.csv`; `failures.csv` | numerical gate failure recorded, not omitted |
| Bounded-RBF primary paired deltas and W/T/L | `paired_statistics.csv`; `paired_primary.csv`; `analysis/paired_seed_details.csv` in the Gate8 run | 7 Blood pairs, 8 others; screening only |
| Mechanism-analysis coefficient shrinkage, support reduction, slack increases | `results/robust/bounded_rbf_redesign/r10_failure_analysis/paired_objective_deltas.csv`; `dataset_mechanism_summary.csv` | deterministic reconstruction from stored models |
| Mechanism-analysis corrected/new-error counts | `dataset_flip_summary.csv`; `prediction_flips.csv` in the same directory | pooled stored test predictions |
| Threshold-versus-function interpretation | `docs/robust/08_r9_failure_analysis/R9_GATE8_FAILURE_ANALYSIS.md`; `threshold_score_movements.csv` | mechanistic post hoc analysis; no new fitting |
| Robust final scientific conclusion | `docs/robust/ROBUST_RESEARCH_FINAL_REPORT.md`; `ROBUST_RESEARCH_TIMELINE.md`; `ROBUST_PAPER_SUMMARY.md` | closed-track synthesis |
| Seed independence and untouched robust ranges | `docs/robust/03_training_protocol/SEED_REGISTRY.md`; expanded `stage_registry.csv` | authoritative seed registry |
| Project architecture decisions and frozen status | `docs/ARCHITECTURE_HISTORY.md`; `docs/PROJECT_PROGRESS.md` | project-wide research narrative |
| Reference metadata used in bibliography | `docs/SVM.md`; `docs/Reproduction_Report_Robust_SVM.md`; `docs/SVM_Paper_Analysis.md` | repository-available citation metadata; see QA caveat |

## Protocol distinctions enforced in the manuscript

- Literature numbers are not used as paired confirmation evidence.
- Gate8/Gate24 estimates are labeled development or screening evidence.
- The seven deterministic confirmation rows are not enlarged with the two negative RKHS/source-profile Gate8 results.
- Expanded Breast Cancer recurrence is not conflated with Breast Cancer Diagnostic.
- Heart is described as the local five-class protocol.
- The Bounded-RBF Class-Sensitive Robust q1 SVM is described as rejected even though its weighted robust solver completed 32/32 fits.
- No robust Gate24 or 96-seed confirmation result is claimed; those runs were never launched.
