# TSP DRL — Independent-seed extension runbook (v1.2)

## Purpose

The principal experiment remains `clean_restart_seed20261013_v1` plus its independent evaluation `final_evaluation_v1_1`. This extension adds two independent training seeds without modifying those artifacts.

## Fixed experiment contract

- Training seeds: `20261014` and `20261015` (in addition to existing `20261013`).
- Same exact Git commit as the existing clean-restart run; the notebook reads the SHA from its saved `manifest_pre.json`.
- Same architecture and hyperparameters: n=20 during training, batch size 512, embedding/hidden dimensions 128, Adam at 0.001, same learning-rate schedule, 13,000 optimizer updates.
- No warm-start or checkpoint transfer between seeds. Each run begins from a fresh initialization.
- Checkpoint selection independently per run, using only 100 n=20 validation instances, greedy decoding, and the existing rule: minimum mean validation tour length among suffixes 1000, 5000, 10000 and 12999.
- Final comparison on exactly the same fresh test instances across all three seeds: n in {20, 50, 100}, 100 instances per size, final-test base seed 8302026 and effective seeds 8302046/8302076/8302126.
- Classical comparison configuration stays fixed (NN, NN+2-opt, ACO with 15 ants, 30 iterations, alpha=1, beta=3, rho=0.1, q=1).

## Run sequence in Google Colab

1. Open `TSP_DRL_Multiseed_Training_v1_2.ipynb`, select a GPU runtime and run from top to bottom. It trains seeds 20261014 and 20261015 sequentially and records a separate `manifest_training.json`, checkpoint hashes and log for each.
2. Only after both training runs finish and their manifests show 13,000 updates and checkpoints with suffixes 1000, 5000, 10000, and 12999, run `TSP_DRL_Independent_Evaluation_seed20261014_v1_2.ipynb`.
3. Run `TSP_DRL_Independent_Evaluation_seed20261015_v1_2.ipynb` in the **same Colab runtime/session when possible**. Each evaluation selects on validation only, generates the same final-test coordinates after selection, and writes to a distinct output folder. The aggregation script checks exact coordinate identity and matching PyTorch/CUDA/GPU metadata before comparing seed results.
4. Upload `aggregate_multiseed_results.py` to the Colab session, or place it on a machine where the Google Drive project root is available. Run:

   ```bash
   python aggregate_multiseed_results.py --project-root "/content/drive/MyDrive/PFE_TSP_DRL"
   ```

   For another mount/path, pass that path to `--project-root`. By default the script refuses to replace an existing aggregation output folder.

5. Review `multiseed_aggregation_manifest.json`, `multiseed_selected_checkpoints.csv`, `multiseed_per_seed_summary.csv`, `multiseed_across_seed_summary.csv`, and `multiseed_paired_bootstrap_by_seed.csv` before editing the thesis.

## What each output means

- `multiseed_per_seed_summary.csv`: each selected model's mean and standard deviation across the same 100 test instances, separated by n and decoding method.
- `multiseed_across_seed_summary.csv`: mean of the three seed-level means and the sample standard deviation *between training seeds*. This is the important addition; n=3 still supports descriptive, not strong population-level, claims.
- `multiseed_paired_bootstrap_by_seed.csv`: paired uncertainty across test instances conditional on each fixed model. It does not estimate training-seed variability and has no multiplicity correction.
- `multiseed_combined_per_instance.csv`: raw per-instance tours, recomputed empirical gaps, run IDs and training seeds.

## Scientific interpretation rules

- The original seed-20261013 table is preserved. The extension is reported as an added robustness analysis, not a rewritten history.
- Do not describe a gap against the best observed method as an optimality gap; no exact TSP optimum was computed.
- If the three seeds disagree materially, report the disagreement rather than selecting only the best seed. Main comparisons should use the mean across training-seed means and show between-seed SD, alongside per-seed results.
- Testing a model trained only at n=20 on n=50 and n=100 measures transfer to larger instances, not training at those sizes.
- Do not treat the 300 seed-instance outcomes as 300 independent test instances: the same 100 test instances are intentionally paired across seeds.
- Do not claim DRL is generally superior to ACO or 2-opt based on these runs. Conclusions remain limited to this architecture, hyperparameter setting, training budget, baselines and test distribution.
- The existing v1.1 timing protocol is not repeated for all seeds by default. The additional-seed extension focuses on tour quality and seed-to-seed robustness; retain the original timing discussion as implementation-specific.

## If a run fails

Do not overwrite the output folder and do not resume from an actor-only checkpoint. Preserve `training_stdout.log`, all snapshots and manifests; inspect the error and decide whether a separately named controlled rerun is needed. A completed run should have the final actor checkpoint, required selection checkpoints and a matching hash in `manifest_training.json`.

## Thesis status before the extension runs

Until both new runs and their evaluations pass the integrity checks, the thesis must continue to describe the principal results as coming from one training seed (20261013). Do not fill in cross-seed tables or claim that seed variability was measured before the aggregation script has run successfully.
