# Validation report — TSP DRL multi-seed extension

## Static validation completed

- Training notebook: JSON loads successfully; every Python code cell parses with `ast.parse`.
- Evaluation notebook for seed 20261014: JSON loads successfully; every Python code cell parses with `ast.parse`.
- Evaluation notebook for seed 20261015: JSON loads successfully; every Python code cell parses with `ast.parse`.
- Aggregation script: passes `python -m py_compile`.
- Aggregation integration smoke test: passed against synthetic artifacts for three seeds, including tour-permutation checks, route-length recomputation, same-instance checks, cross-seed summaries, and paired bootstrap outputs.

## Not run / not claimed

- No GPU training was launched in this environment.
- No real evaluation for seeds 20261014 or 20261015 has been completed here.
- The synthetic test validates the aggregation script's structure and arithmetic only; it is not scientific evidence about the trained models.
- The existing thesis source and existing v1.1 result tables were not modified.

## Expected real outputs

The runbook defines the required Drive paths and exact notebook execution order. The aggregation script refuses to combine outputs unless it finds all three evaluation folders, the same validation/test coordinate arrays, consistent classical baseline tours and lengths, and matching PyTorch/CUDA/GPU inference metadata. The new evaluation notebooks reuse the original timing profile instead of repeating timing measurements.
