# Evaluation protocols for ML-based water leak detection

Code, extracted features and per-fold results for the paper:

> **Evaluation Protocol, Not Model Choice, Determines Reported Accuracy in Acoustic Leak
> Detection: Evidence from Three Sensor Modalities**
> AbdelHafiz Itani and Mohammad Watfa, Lebanese American University.

The study holds features, classifier and hyperparameters fixed and varies **only the rule by
which data are split into training and test sets**, on the public leak-detection benchmark of
Aghashahi, Sela and Banks (2023).

## Headline result

Balanced accuracy, Random Forest, mean over five seeds:

| Protocol | Hydrophone | Accelerometer | Dyn. pressure |
|---|---|---|---|
| **Binary** — P1 random window | 0.923 | 0.984 | 0.851 |
| **Binary** — P4 cross-topology | 0.553 | 0.530 | 0.571 |
| **5-class** — P1 random window | 0.946 | 0.966 | 0.833 |
| **5-class** — P4 cross-topology | 0.120 | 0.261 | 0.189 |

Chance is 0.500 (binary) and 0.200 (five-class).

## Protocols

| ID | Rule |
|---|---|
| P1 | Stratified 5-fold CV over **windows**; windows of one recording may fall on both sides |
| P2 | 5-fold grouped CV; no **recording** spans the train/test boundary |
| P2b | As P2, but the two sensors of one **experimental run** are kept together |
| P3 | Each background flow/noise **condition** held out in turn |
| P4a / P4b | Train on one network **topology**, test on the other (both directions) |

## Repository layout

```
code/
  Aghashahi_feature_extraction_COLAB.ipynb  Colab notebook: raw signals -> feature CSVs
  make_spectrograms.py                      log-mel spectrogram patches (hydrophone)
  protocol_v2.py        main protocol study (RF, 3 modalities, 2 tasks, 6 protocols, 5 seeds)
  protocol_all.py       secondary study (RF + SVM, amplitude-invariant feature subset)
  spec_protocol.py      protocol study on learned spectrogram representation
  remedies.py           normalisation remedies + transfer diagnostics + per-class recall
  make_figs.py          figures 1-4
  make_figs2.py         figures 5-6
data/features/          per-window features (45 per window) for the three modalities
results/                per-fold results; every number in the paper comes from these
figures/                figures as they appear in the paper
```

## Reproducing

The raw signals are **not** redistributed here. Download them from Mendeley Data
(doi:10.17632/tbrnp6vrnj.1) and run the Colab notebook to regenerate `data/features/`.
Everything downstream runs from the feature CSVs already in this repository.

The scripts read `feats/<modality>_features.csv` relative to the working directory, so link
that name to `data/features/` once and run them from the repository root:

```bash
pip install -r requirements.txt
ln -s data/features feats        # or: mkdir feats && cp data/features/*.csv feats/
mkdir -p figs

python code/protocol_v2.py      # -> protocol_v2.csv
python code/protocol_all.py     # -> protocol_all.csv
python code/remedies.py         # -> remedy_results.csv, perclass_results.csv
python code/make_spectrograms.py && python code/spec_protocol.py   # needs raw hydrophone .raw
python code/make_figs.py && python code/make_figs2.py              # -> figs/
```

Reference copies of all five result tables are in `results/`; the commands above write fresh
ones to the working directory so you can diff them against the published values. All random
seeds are fixed (0-4), so reruns reproduce the published numbers exactly on the same library
versions (see `requirements.txt`).

Runtime on a single CPU core is roughly 25 minutes for `protocol_v2.py` and under 10 minutes
for each of the others.

## Data

Aghashahi, M., Sela, L., & Banks, M. K. (2023). Benchmarking dataset for leak detection and
localization in water distribution systems. *Data in Brief*, 48, 109148.
https://doi.org/10.1016/j.dib.2023.109148

The accelerometer and dynamic pressure CSV files carry a time column corresponding to
**25.6 kHz**; the feature extraction infers the rate from that column rather than assuming it.

## License

Code and derived results: MIT (see `LICENSE`). The underlying measurements remain under the
licence of the original dataset.

## Citation

See `CITATION.cff`, or use the "Cite this repository" button on the GitHub sidebar.
