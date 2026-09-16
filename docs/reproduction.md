# Reproduction guide

Prepare ESC-50, the model-specific vendor code, and pretrained weights separately. Place them according to a private copy of `configs/strict.yaml`; do not commit these assets.

The main commands are exposed through the `trainingless` CLI. `run-table1` evaluates conventional and proposed filtering under the declared corruption settings. `run-figure7` sweeps the ending block. `run-sensitivity` explores the declared cutoff, noise-reference, mask-boundary, and robustness settings. Use `--resume` to reuse generated prediction files.

The proposed path is read-only at inference time: it caches silent activations once, enumerates frequency boundaries, computes information maximization on unlabeled calibration data, and applies the selected boundary without gradients or parameter updates.
