# Contributing

1. Create a focused branch and describe the scientific or engineering purpose.
2. Do not modify frozen model definitions, seed registries, or immutable result
   artifacts without an explicit new protocol and migration note.
3. Fit preprocessing only on training data and use matched splits for paired
   model comparisons.
4. Run `python -m pytest -q` and a one-seed local smoke before opening a pull
   request.
5. Preserve negative results and solver failures; never delete them to improve
   a summary.
6. Do not add cloud-provider dependencies to the active package or CI.

Pull requests should state whether they change mathematics, experiment
semantics, data provenance, or presentation only.
