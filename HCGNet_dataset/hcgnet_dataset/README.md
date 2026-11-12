# Transpilation-cost surrogate (HCG-Net) — code, data and results

Code and data for the paper *"Learning to Predict Transpilation Cost: A Graph Neural Network Surrogate for
Post-Routing Depth, Two-Qubit Gate Count and Estimated Fidelity"*. Every number in the paper is read from
`results/nums.json`, which `make_nums.py` builds from the raw result files in `results/`.

## Environment
Python 3.13.18, qiskit 1.4.6, qiskit-ibm-runtime 0.44.1 (fake back-ends FakeSherbrooke / FakeTorino), torch 2.15.0 (CPU),
scikit-learn 1.9.1, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2. Node.js 22 + `docx` 9.8.1 (npm) only to rebuild the .docx.
All experiments ran on 2 CPU cores, no GPU.

## Files
| File | Purpose |
|---|---|
| `gen_dataset.py` | Generates the 9 circuit families, canonicalises them, transpiles with the Qiskit preset pass manager (opt. level 1–3) on a calibrated target with faulty couplers removed, records depth, two-qubit count, ln ESP and wall-clock time. 5 % of ID circuits get 5 seeds (noise floor). |
| `features.py` | Heterogeneous circuit graph (2q-block nodes, qubit nodes, sequence/incidence edges) and 23 global descriptors. |
| `prepare.py` | Builds graphs for every labelled circuit → `data/<dev>/processed.pkl`. |
| `model.py` | HCG-Net (pure PyTorch). |
| `train.py` | All experiments: `--exp iid` (5 splits, baselines, HCG-Ens), `timing`, `ablation`, `lofo`, `torino` (incl. transfer). |
| `analysis.py` | Dataset statistics and seed-noise floor. |
| `bias_check.py` | Seed-0 mean log-error (bias) on ID and size-OOD pools → `results/bias_s0.json`. |
| `significance.py` | Paired Wilcoxon tests and bootstrap CIs (seed-0 split) → `results/significance_s0.json`. |
| `fig_arch.py`, `fig_parity.py`, `fig_size.py` | Figures 1–3 (`fig_parity.py` also writes `results/breakdown_s0.json`). |
| `make_nums.py` | Aggregates `results/*.json` into `results/nums.json` (tables + every quoted number). |
| `export_data.py` | Writes the dataset as `*_labels.csv` + `*_circuits.qpy`. |
| `import_data.py` | Rebuilds the shard files used by `prepare.py` from the released CSV + QPY (round-trip verified bit-for-bit; retraining seed 0 from the released data reproduces every reported seed-0 metric exactly). |
| `paper/` | docx-js builder for the IEEE A4 paper (`node paper/build.js paper.docx`). |

## Reproduce
From the released dataset instead of regenerating labels: `python import_data.py <dataset_dir> sherbrooke data/sherbrooke` (same for `torino`), then start at `prepare.py`.

```bash
python gen_dataset.py --backend sherbrooke --out data/sherbrooke --n_id 6300 --n_ood 900
python gen_dataset.py --backend torino --out data/torino --n_id 2700 --n_ood 300 --seed 2027
python prepare.py data/sherbrooke sherbrooke && python prepare.py data/torino torino
python analysis.py
python train.py --exp iid --seeds 5 && python train.py --exp timing
python train.py --exp ablation --seeds 3 && python train.py --exp lofo && python train.py --exp torino --seeds 3
python bias_check.py && python significance.py && python fig_size.py && python fig_parity.py && python fig_arch.py
python make_nums.py && STRICT=1 node paper/build.js paper.docx
```
Transpilation times depend on the machine; SABRE is seeded, so labels are reproducible for a fixed Qiskit version.

## Dataset (dataset zip)
`<dev>_circuits.qpy` holds the logical circuits (basis {u, cx}, no measurements) in the same order as the rows
of `<dev>_labels.csv`. Columns: `uid`, `family`, `n_qubits`, `split_pool` (id = 4–20 qubits, ood = 21–28),
`opt_level`, `depth` (incl. final measurement), `n_2q`, `ln_esp`, `esp`, `transpile_s`, `seed`,
`extra_seed_labels` (JSON list of [depth, n_2q, ln_esp] for the additional seeds of noise-floor circuits),
`generator_meta`. Devices: FakeSherbrooke (Eagle r3) restricted to 122 usable qubits; FakeTorino (Heron r1)
restricted to 129 usable qubits (couplers with error ≥ 0.5 removed, largest connected component kept).

## Trained models and logs
`results/hcgnet_sherbrooke_s0.pt` and `results/hcgnet_torino_s0.pt` hold the seed-0 HCG-Net weights with the descriptor scaler and target statistics; `logs/` contains the console logs of the reported runs. Other seeds are retrained by `train.py` (they are not stored).

## Reproducibility notes
* Numbers: `make_nums.py` regenerates every number and table of the paper from `results/*.json`.
* Labels: transpilation is seeded (`seed` column), so re-transpiling a released circuit with Qiskit 1.4.6 gives the same depth, two-qubit count and ESP.
* Models: training is deterministic on CPU with `TORCH_THREADS=1` (the default in `train.py`); we verified that seed 0 retrained from the released data matches the paper exactly for all models. Other library versions, multi-threading or GPUs can change the last digits of HCG-Net results.
* Timing (transpile seconds, inference ms, speed-ups) depends on the machine and will differ.
