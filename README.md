# ICASSP-wls: Frozen Model-Effective Boundary Selection

Code accompanying an ICASSP submission on trainingless adaptation for environmental sound classification. The implementation evaluates a frozen transformer by replacing high-frequency intermediate tokens with cached silent-path activations, then selects a model-effective boundary from unlabeled calibration clips with information maximization.

The repository contains only the method implementation and reproducibility entry points. ESC-50 audio, pretrained weights, checkpoints, and third-party model sources are not included.

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[test]"
```

Set `project_root: .` (or an absolute path) in a copied config, then provide the dataset, vendor sources, and weights at the configured locations.

## Run

```bash
trainingless prepare --config configs/strict.yaml
trainingless run-table1 --config configs/strict.yaml --device cuda
trainingless run-figure7 --config configs/strict.yaml --device cuda
trainingless run-sensitivity --config configs/strict.yaml --device cuda
pytest
```

The `trainingless_adaptation.adaptation` module is the core contribution: `TokenGrid`, `replace_high_frequency_tokens`, `capture_silent_outputs`, and `replacement_hooks` implement the frozen silent-path replacement described in the paper. Model adapters expose the same operation for PaSST, MS-CLAP, and BEATs. The experiment runner provides Table I, Fig. 7, sensitivity analysis, and cross-fitted selection workflows.

The `trainingless_adaptation.selection` module contains the label-free selector used by the paper: probability-tensor validation, entropy and IM scores, paired bootstrap selection, largest-boundary tie handling, and evaluation helpers that keep labels outside the selector interface.

## Reproducibility notes

The code does not claim that every published number is reproduced automatically. Results depend on the exact pretrained weights, vendor revisions, dataset files, CUDA/PyTorch versions, and configuration choices. Keep generated outputs outside Git and record local paths in a private config.

## License

Released under the MIT License. The datasets, pretrained models, and vendor implementations retain their original licenses.
