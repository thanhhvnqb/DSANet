# Research project rules

@/home/islab/.codex/RTK.md

- Always use English in answers, code, comments, documentation, logs, commit messages, and reports.
- Prefix every shell command with `rtk`, using `rtk proxy` for unsupported commands.
- Preserve existing user changes and historical experiment artifacts. Never invent results or silently overwrite runs.
- When experiments are needed, prepare and check the code and provide exact commands for the user to run. Do not launch training, full evaluation, sweeps, or GPU experiments unless explicitly requested. Lightweight code checks are allowed.
- Store new experiment outputs in `runs/<hypothesis>/<variant>/<dataset>/seed_<seed>/`. Use meaningful hypothesis and variant names; identify the method in the variant. Dataset names are `ucf` and `xd`.
- Keep training logs, test logs, best inference weights, and best training checkpoints together in that directory. Save the best model immediately upon selection improvement. Use a new hypothesis/variant for repeat runs with the same seed; never overwrite an earlier run implicitly.
- Record configuration, seed, command, code revision, environment, selection split, selection metric, selection event, and checkpoint path. Preserve metric units and branch definitions.
- In experimental summary and comparison tables, put the original-paper and reproduced VadCLIP and DSANet reference rows first, before proposed variants. Include all four reference rows; mark unavailable values explicitly rather than inventing them. Diagnostic/training tables without comparable paper values must state this clearly.
- Compare every proposed result against both methods' paper values and reproduced values. Report absolute percentage-point differences, metric direction, dataset, branch, seed count, and evaluation protocol.
- Cite primary papers with table numbers and link reproduced results to their logs. Keep paper claims separate from local measurements. Use `reports/performance_report.md` as the initial baseline reference.
- Disclose test-set checkpoint selection, prompt/preprocessing changes, missing measurements, and differences between selection-time and standalone evaluation. Do not claim statistical significance from a single seed.
- Preserve baseline behavior when implementing hypotheses. Make one controlled change per variant and document deviations. Keep train/test prompts, preprocessing, and checkpoint architecture consistent.
- Validate code with appropriate lightweight checks. Report what was checked and what requires user-run experiments. Do not commit datasets, secrets, model weights, or large run artifacts.
