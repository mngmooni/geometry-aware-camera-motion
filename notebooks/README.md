# Training and evaluation notebooks

Use `dataset_and_evaluator.ipynb` to check the synthetic dataset and evaluator, then the model notebooks: `fm_rel_pose.ipynb`, `fm_rel_inc.ipynb`, and `rfm_so3.ipynb`.

The notebooks preserve the original training and comparison workflows. They accept a Git clone URL (`REPO_URL` in the first code cell) or a repository ZIP upload in Google Colab; local execution starts from the repository root. Training/evaluation outputs were written to Google Drive in the original experiments.

Some comparison cells expect existing checkpoints or previous experiment directories. The historical, unmatched absolute-pose exploration is retained under `archive/`, but it is not part of the three-model matched benchmark. To start a fresh model run without those previous checkpoints, use `python scripts/train_fm.py --help` and `python scripts/eval_fm.py --help`.

Notebook-builder scripts were deliberately omitted from the public artifact: the already-generated notebooks are the published executable sources.
