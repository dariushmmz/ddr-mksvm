# Expanded dataset study — paper-ready summary

The full local dataset directory was audited: nine classifier datasets are
scientifically usable under documented protocols, while one CSV is metadata
rather than a learning dataset. Seven datasets reuse prior immutable evidence;
Breast Cancer recurrence and Dermatology underwent a prospectively staged
smoke/Gate8/Gate24/confirmation sequence.

All new stages completed without numerical failure. The untouched 96-seed
confirmation found that V7 reduced Breast recurrence error from `30.64%` to
`26.81%` (paired delta `−3.83` pp, 95% CI `[−4.71,−2.95]`, W/T/L 80/4/12)
and reduced Dermatology error from `4.27%` to `3.81%` (delta `−0.46` pp, CI
`[−0.90,−0.02]`, W/T/L 46/21/29). V7 was approximately `1.70x` and `9.02x`
faster, respectively.

These are not unqualified wins. Breast recurrence minority recall fell by
`8.43` pp and Dermatology rare class-6 recall fell by `2.92` pp. Balanced
accuracy and macro-F1 did not materially improve. V8-C produced interesting
balance gains on Breast at Gate24 but had already failed the frozen Gate8 error
guard and again exceeded the Gate24 error guard; Dermatology V8-C was nearly
identical to V7 but slower. It therefore remained a fixed control and did not
consume confirmation seeds.

For the paper, all nine usable datasets should remain visible. The strongest
V7 evidence is Parkinson, Breast Cancer Diagnostic, Wine, Iris, Breast
recurrence, and the smaller Dermatology result. Blood and Mammographic remain
prospective V7 negatives. V8-C's strongest supported class-sensitive evidence
remains Blood and Heart, with explicit error/runtime tradeoffs. The expanded
results strengthen both external validity and the limitations section: V7 is
an effective accuracy/runtime reference, but it does not guarantee minority-
class protection.

## Recommended manuscript exhibits

1. **Dataset/protocol table:** nine included datasets, sample/features/classes,
   preprocessing, missing-data handling, and exact confirmation registry.
2. **Legacy versus V7 table:** error, balanced accuracy, macro-F1, paired error
   CI/W-T-L/effect, and runtime ratio for the seven confirmed datasets; Blood
   and Mammographic appear as Gate8 negatives in the same exhibit or footnote.
3. **V7 versus V8-C table:** the seven confirmations, highlighting Blood's
   balance gain and Heart's balance/error/runtime tradeoff; Breast recurrence
   and Dermatology appear as stopped Gate24 controls.
4. **Class-recall figure:** paired minority/rare-class recall deltas, with
   Blood's V8-C gain and Breast recurrence/Dermatology V7 losses visible.
5. **Runtime figure:** log-scale runtime ratios for both comparisons.
6. **Gate flow figure:** smoke → Gate8 → Gate24 → confirmation, including
   prospective pruning and preserved negative branches.

Exact manuscript citations should point to the four immutable source runs
listed in `results/expanded_dataset_study/analysis/cross_dataset/provenance.json`
and to their original paired-comparison files, not to development means.
