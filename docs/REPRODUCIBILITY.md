# Reproducibility and evaluation notes

## Setup

Use Python **3.10, 3.11, or 3.12** and the packages listed in `requirements.txt` (PyTorch >=2.1,<3). GitHub Actions runs these versions on Linux. Package versions are constrained to compatible major-version ranges rather than frozen to a complete lockfile; identical results across systems or future dependency releases are not guaranteed. No model checkpoints are distributed.

```bash
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider
python scripts/run_gt_audit.py --out outputs/phase1
```

To run the SO(3) diagnostic without rerunning an entire training:

```bash
python scripts/diagnose_so3_transport.py --n 40 --n-train 200 --out outputs/diagnostics/so3_transport
# Complete diagnostic from frozen synthetic training data (slower):
python scripts/diagnose_so3_transport.py --out outputs/diagnostics/so3_transport
```

To attach saved solver sweeps, add `--sweep rfm_so3_E1b=path/to/solver_sweep_eval_id.json` (repeat for additional available sweeps). Source metrics and final diagnostic outputs are archived in `results/diagnostics/so3_transport/`.

## Training/evaluation workflows

The Colab notebooks in `notebooks/` preserve the model training and evaluation workflows. In order: `dataset_and_evaluator.ipynb` (check), `archive/fm_abs_pose_exploratory.ipynb` (historical), `fm_rel_pose.ipynb`, `fm_rel_inc.ipynb`, and `rfm_so3.ipynb`. **Historical absolute-pose evaluations used a different evaluator and cannot be presented as a matched contemporary comparison.** The original absolute-pose checkpoint was not preserved and is not required for the published main three-method benchmark.

For new runs:

```bash
python scripts/train_fm.py --help
python scripts/eval_fm.py --help
python scripts/make_benchmark.py --help
python scripts/render_demo.py --help
```

Some notebook cells assume Colab Google Drive paths; change these to your working directories. For an exact end-to-end *sample regeneration*, you must retrain or supply your own compatible trained checkpoints, since the historical `.pt` files are not included.

## Archived evaluation

The main tables and figures are derived from the preserved relative-pose, increment and SO(3) experiment outputs; the transport diagnostic was run separately. `results/benchmark/data/benchmark_table_original.csv` is preserved without editing. `results/benchmark/data/main_results_annotated.csv` adds a status field for unmatched E1b jerk references. Paired CI details are archived as text, not recomputed. We do **not** infer model-seed uncertainty from held-out-condition resampling.

## Media production

The original MP4s in `results/demo/` come from the project's software camera renderer. The two `*_three_model_synchronized.mp4` videos were compiled from existing MP4s with time-aligned resizing, panel labels and juxtaposition only. They use independent samples from the same request, not matched output trajectories. No additional inference or rerendering was done during this final packaging.

## Scope of reproducibility

- **Available immediately:** code, synthetic data generator, test suite, frozen benchmark tables, existing figures and videos, and no-training SO(3) diagnostics.
- **Requires retraining:** reproducing newly sampled trajectories from the original fitted models, since checkpoints were not preserved.
- **Not a cross-platform lockstep reproduction:** exact environment lockfile, model weights, and multi-seed replication are not provided.
- The untrained correlated Haar counterfactual is an analytical diagnostic, not a trained baseline. Single-seed and E1b jerk-reference limitations are documented in the [research report](RESEARCH_REPORT.md).
