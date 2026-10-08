# Robust-method research track

**Final status (2026-09-19): scientifically closed. No robust candidate is
promoted and no robust experiment is authorized.** The authoritative synthesis
is the [final research report](ROBUST_RESEARCH_FINAL_REPORT.md); the concise
sequence is in the [research timeline](ROBUST_RESEARCH_TIMELINE.md), and the
publication-ready account is in the [paper summary](ROBUST_PAPER_SUMMARY.md).

This directory is the entry point for the auditable robust-method track. The
RKHS-L2 Robust V8-C SOCP is a closed historical candidate, not the current
candidate. R3.3 ended its numerical-rescue path after demonstrating effective
Gram conditioning above `1e23` on the exposed Blood outer problems.

That candidate is **not frozen**. The 24-seed development gate stopped after
repeatable CLARABEL exceptions on Blood Transfusion seeds 15000 and 15001.
Parkinson's small favorable metric shifts were inconclusive, runtime was
21.14 times the V8-C reference, and 18/24 selected robust outer models were
reported `optimal_inaccurate`.

Historical R3 solver qualification was **NOT YET QUALIFIED**: exact duplicate
aggregation fixed all inner-call exceptions, but both selected Blood outer fits
and independent high-accuracy CVXOPT failed. R3.3 now supersedes further solver
diagnostics and closes numerical rescue of this SOCP. R4, a new four-dataset
gate, and the untouched 96-seed confirmation remain blocked.

R3.2 tested the requested exact RKHS-coordinate transformation and stopped at
its mandatory factor-existence gate: neither stored Blood outer Gram matrix
admits an unmodified full-rank Cholesky factor. No R3.2 optimization was run.
Decision: **R3.2 NOT QUALIFIED**.

R3.3 reconstructed both kernels with certified arbitrary-precision interval
arithmetic. Each is mathematically SPD at 2048 bits, but estimated condition
numbers are `7.11e23` and `5.01e24`, and neither passes the frozen requirement
for stable factorization at two consecutive precision levels. No solver was
invoked. Numerical rescue of the current RKHS-L2 SOCP is closed; that result
led to the now-completed R8 source-aligned redesign. R4–R7 remain closed.

R8 derived one candidate from the
authoritative original `q=1` robust LP: mean-one square-root inverse-frequency
slack weights, with every robust constraint and source prediction semantic
retained. The candidate remains an LP and exactly reduces to the original
robust model when weighting is disabled. Implementation invariants pass, but
exposed-seed qualification is **NOT QUALIFIED**: both Blood seeds fail all
four q1 variants under the frozen HiGHS policy. Parkinson solves cleanly, but
cannot unlock a gate without Blood. No fresh R8 seed has been consumed.

R8.1 then tested one exact positive-diagonal LP equilibration rule. Parkinson
equivalence passed, but only 5/8 exposed Blood models completed; three
deterministic variants still returned HiGHS status 4. Decision: **R8.1 NOT
QUALIFIED**. Further solver/scaling rescue of this Blood/source-profile path is
closed, and no fresh seed has been consumed.

R9 was the latest candidate. It replaces the closed Blood polynomial kernel
with one prospectively frozen Gaussian RBF while retaining the q1 robust LP,
`p=2`, `rho=.01`, source selection/threshold semantics, and `tau=.5` slack
weights. Exposed numerical qualification passed: all eight Blood models and
four Parkinson controls solved, with maximum Blood violation `6.63e-8` and
maximum canonical coefficient `1.0`. One unweighted robust Blood result was
majority-only, so exposed qualification established numerical feasibility only.

The subsequent fresh Gate8 on 15300–15307 **failed**. Blood seed 15305 failed
both deterministic controls with HiGHS status 4, leaving 126/128 tasks and
31/32 primary pairs. The available pairs were adverse overall: mean delta
error `+.01207`, balanced accuracy `-.01930`, macro F1 `-.01861`, and safety
recall `-.04407`. Gate24 and confirmation remain untouched and blocked. No
R9.1 rescue or parameter search is authorized.

R10 closes the diagnosis using only stored Gate8 checkpoints. It identifies
global coefficient shrinkage/slack substitution from the robust constraint,
followed by an unweighted source threshold that favors the majority on the
two clearest safety-recall failures. R9 is a prospective negative result, not
a candidate awaiting tuning. Gate24 and confirmation are permanently blocked
for frozen R9, and the current robust track is closed. Only a future,
independently derived formulation could begin a separate robust project.

## Index

- [Authoritative final research report](ROBUST_RESEARCH_FINAL_REPORT.md)
- [Complete research timeline](ROBUST_RESEARCH_TIMELINE.md)
- [Publication-ready summary](ROBUST_PAPER_SUMMARY.md)
- [Overview and current status](00_overview/ROBUST_EXPERIMENT_STATUS.md)
- [Research roadmap](00_overview/ROBUST_ROADMAP.md)
- [Stopped Gate24 evidence](01_baseline_and_gate24/GATE24_SUMMARY.md)
- [Solver qualification](02_solver_qualification/SOLVER_QUALIFICATION_PLAN.md)
- [Independent CVXOPT qualification](02_solver_qualification/INDEPENDENT_SOLVER_QUALIFICATION.md)
- [R3.2 RKHS-coordinate reformulation](02_solver_qualification/RKHS_COORDINATE_REFORMULATION.md)
- [R3.3 high-precision kernel qualification](02_solver_qualification/HIGH_PRECISION_KERNEL_QUALIFICATION.md)
- [Training protocol](03_training_protocol/ROBUST_TRAINING_PROTOCOL.md)
- [Seed registry](03_training_protocol/SEED_REGISTRY.md)
- [Future gate protocol](04_future_gates/NEXT_GATE_PLAN.md)
- [Machine-readable result map](05_results/README.md)
- [R8 class-sensitive q1 LP design](06_source_aligned_redesign/ROBUST_CLASS_SENSITIVE_LP_DESIGN.md)
- [R8 prospective experiment plan](06_source_aligned_redesign/EXPERIMENT_PLAN.md)
- [R8 experiment log](06_source_aligned_redesign/EXPERIMENT_LOG.md)
- [R8 smoke status](06_source_aligned_redesign/SMOKE_REPORT.md)
- [R8.1 exact LP equilibration](06_source_aligned_redesign/LP_EQUILIBRATION_QUALIFICATION.md)
- [R9 bounded-RBF design](07_bounded_rbf_redesign/ROBUST_RBF_Q1_DESIGN.md)
- [R9 exposed qualification](07_bounded_rbf_redesign/SMOKE_REPORT.md)
- [R9 fresh Gate8 protocol](07_bounded_rbf_redesign/GATE8_PROTOCOL.md)
- [R9 fresh Gate8 result](07_bounded_rbf_redesign/GATE8_REPORT.md)
- [R10 R9 failure mechanism analysis](08_r9_failure_analysis/R9_GATE8_FAILURE_ANALYSIS.md)

## Current authorization

The current robust research track is closed. R8/R8.1 are closed, and R9 Gate8
failed and is not promoted. There is no presently authorized robust experiment.
Gate24 and confirmation were not launched. A future robust project would
require an independently derived formulation and a new dated protocol before
any preserved seed is accessed; it is not a continuation or rescue of a closed
candidate.

## Prohibited after failed R9 Gate8

- any use of seeds 14000–14095;
- the untouched 96-seed confirmation;
- any use of 15400–15423;
- reuse of consumed 15300–15307 as fresh or confirmation evidence;
- searches over `p`, `rho`, kernels, regularizers, feature pipelines, class
  weights, or other predictive architecture choices;
- ad-hoc tolerance or solver changes during a scientific gate.

Seeds 15400–15423 and 14000–14095 are untouched and preserved for a future
independently derived method. Their preservation is not authorization to use
them.

Frozen V7, frozen V8-C, and all deterministic artifacts remain immutable.
