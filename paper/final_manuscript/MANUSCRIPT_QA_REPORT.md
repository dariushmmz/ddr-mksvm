# Manuscript quality-assurance report

Date: 2026-09-27  
Canonical source: `SVM_FINAL_PAPER.md`

## Outcome

The revised manuscript package passed its evidence, terminology, mathematics, citation, rendering, regression, and frozen-source checks. The DOCX and PDF were generated from the same Markdown source. Post-conversion processing changes only typography, pagination, headers, footers, equation layout, and table/figure presentation.

No model was fitted and no experiment or machine-readable scientific result was changed during this revision.

## Checks completed

### Scientific and numerical integrity

- Rechecked headline deterministic summaries against `results/expanded_dataset_study/analysis/cross_dataset/confirmed_paired_statistics.csv` rather than copying rounded prose.
- Verified seven Source-Profile q1 versus RKHS-RBF-OVO and seven RKHS-RBF-OVO versus CS-RKHS-RBF-OVO untouched 96-seed confirmations.
- Verified every RKHS-RBF-OVO confirmation error interval in Table 4 lies below zero.
- Verified the Parkinson headline values (14.243% versus 7.079%) and Blood class-sensitive balanced-accuracy change (+5.820 percentage points).
- Verified that the Bounded-RBF robust balanced-accuracy deltas are negative on all four fresh Gate8 datasets.
- Checked robust Gate8 pair counts: Blood has seven matched pairs because two deterministic controls failed at seed 15305; the other datasets have eight.
- Kept Gate8/Gate24 evidence separate from untouched confirmation evidence.
- Verified that the Limitations section and the complete manuscript package contain no dataset-unavailability claim.
- Retained Blood and Mammographic negative screening results, recurrence and Dermatology rare-class costs, null or adverse class-sensitive controls, the rejected dual-kernel investigation, every robust numerical failure, and the rejected bounded-RBF robust Gate8.
- Confirmed that no robust Gate24 or 96-seed confirmation result is claimed.
- Confirmed the frozen implementation hashes remain unchanged:
  - RKHS-RBF-OVO SVM (`ddr_mksvm/v7_cross_dataset.py`): `39a2a2c53df267bcb731f1c22021af19518a43af030e6e4b2b88dc1c6746e06d`
  - CS-RKHS-RBF-OVO SVM (`ddr_mksvm/v8_class_sensitive.py`): `fd17cce0f7a2b19f64fe3d085510812f71338a901a3731ec686cbc917100b448`

### Terminology and mathematical presentation

- Replaced internal development labels with descriptive scientific architecture names throughout the narrative, equations, tables, figures, captions, discussion, and conclusion.
- Confined internal labels to the single architecture-mapping table needed to connect the article to immutable code and result artifacts.
- Verified 14 numbered equations, consistent RKHS/kernel/weight/uncertainty notation, and in-text equation references. The numbers are visible and right aligned in both DOCX and PDF.
- Verified that inline mathematics remains within justified prose and only display equations are centered in the DOCX/PDF.
- Verified the supported architecture subsections cover preprocessing, kernel, objective, regularization, class weighting, decomposition, model selection, decision rule, and the mathematical distinction from the source-profile baseline.

### Automated manuscript validation

`validate_manuscript.py` completed 28/28 checks:

- all required sections present;
- internal architecture labels confined to Table 1;
- all 14 numbered equations present;
- all 19 references cited and every citation resolved;
- five figure links present and resolvable;
- prohibited acceptance and universal-superiority claims absent;
- confirmation counts and selected headline values match machine-readable results;
- frozen implementation hashes match;
- the DOCX contains 12 publication tables and five figures;
- the DOCX visibly numbers display equations (1)-(14);
- Section 5.5 contains no conversion-only empty paragraphs;
- the PDF contains the complete conclusion and all five figure captions.

### Software regression

The complete repository test suite passed:

```text
184 passed, 1 third-party pytz deprecation warning
```

Third-party pytest plugin autoload was disabled because of the previously documented unrelated Hydra/ANTLR collection conflict. No training command was run.

### Language and presentation

- Microsoft Word reported zero spelling errors and zero grammatical errors on the final DOCX.
- The 17-page PDF was visually inspected across the title/abstract, equations, all table and figure regions, Section 5.5, discussion, limitations, references, and declarations.
- Section 5.5 now has ordinary heading-to-content spacing, no empty paragraphs, compact provenance text, and no abnormal vertical gap.
- The body prose is justified, display equations are centered, running headers and page numbers are consistent, and dense pages preserve the footer/body separation.
- No broken table, horizontal overflow, clipped figure, blank trailing page, or orphaned terminal heading was found.
- The final package contains 12 publication tables, five 300 dpi figures, matching SVG figure sources, live equations, and 19 references.
- Document properties contain the manuscript title, subject, and keywords.

## Output fingerprints

| File | SHA-256 |
|---|---|
| `SVM_FINAL_PAPER.md` | `3be46d28de535b1ce955c5f89c25b2bfae35c33dc5a89b9c07b6296c9c058751` |
| `SVM_FINAL_PAPER.docx` | `5e3a8611ecb21022859a183d5dade5caa453bfb46a67b410b05308dafbda33a7` |
| `SVM_FINAL_PAPER.pdf` | `921fb9be6fe0c1e23edbfe47928868ba2fc4589df8757ed9fedf871fa11a7df4` |
| `PAPER_DATA_TABLES.md` | `1746c3881fee5608c9057fdf40b7039d99e7efe8c8fd3293e9ad17c1115eb351` |

## Unresolved items before submission

1. Add final author names, affiliations, ORCID identifiers, corresponding-author details, funding, conflicts, author contributions, and the target journal's AI-use statement.
2. Select a journal and apply its exact template, word-count, reference, figure, and supplementary-material requirements.
3. Add a permanent public repository/archive identifier and verify dataset redistribution licenses.
4. Before submission, manually verify every bibliographic record against the publisher or index and complete a current systematic literature search. Missing DOI or issue metadata was not invented.
5. Decide whether the local five-class Heart protocol belongs in the target journal's main text or supplement.
6. Decide whether the detailed robust numerical qualification belongs in the main article or a dedicated supplement; the current manuscript retains a concise scientific synthesis in the main text.
7. Repeated holdouts, lack of external institutional cohorts, and environment-specific runtime comparisons remain disclosed limitations.

## Rebuild commands

From the project root:

```powershell
python paper/final_manuscript/generate_paper_figures.py
python paper/final_manuscript/build_paper_data_tables.py
python paper/final_manuscript/build_manuscript_outputs.py
python paper/final_manuscript/validate_manuscript.py
```
