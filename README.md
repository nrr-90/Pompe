# MultiCausGRN Pompe — pre-publication review package

**Status: NOT YET VERIFIED FOR PUBLIC RELEASE.** This package preserves the supplied model source, original Colab analysis exports, and run metrics. It is an audit baseline, not a runnable end-to-end workflow.

## Contents
- `src/`: original MultiCausGRN source files from uploaded Code ZIP (unchanged).
- `notebooks/`: uploaded primary GSE159062 Colab notebook (unchanged).
- `analysis/`: uploaded GSE38680, manuscript analysis, and perturbation exploratory Colab exports (unchanged; many `!` commands are notebook-only).
- `results/metrics/`: three uploaded run-level metrics files.
- `docs/metrics_review.md`: side-by-side run metrics.

## Data provenance
- GSE159062 and GSE38680: NCBI GEO. Obtain source data directly from the original archive.
- TRRUST: external regulatory interaction prior; record exact release/version.
- Norman 2019, Dixit 2016, Frangieh 2021: external perturbation datasets; source access instructions and versions still need verification.

## Required work before public release
1. Replace personal Colab paths and notebook shell commands with parameterized entry points.
2. Separate exploratory cells from final analyses; document exact execution order and output files.
3. Verify raw-data processing, sample labels, splits, TRRUST reference, model checkpoints and predictions.
4. Reproduce all manuscript tables/figures and test/validation metrics using fixed environments.
5. Add version-pinned Python and R/Bioconductor dependencies and a verified license.
6. Inspect notebook metadata/outputs and paths for personal information before publishing.
7. Verify original data redistribution permissions and archive an immutable release.

Do not cite this package as a public repository until the verification steps are completed.
