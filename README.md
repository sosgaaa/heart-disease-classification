# Heart Disease Classification 🫀

Three models, 297 records, and a healthy respect for cross-validation.

A machine learning project built at **EPFL for Introduction to Machine Learning (CS-233)** with **Othmane Housni and Ali El Azdi**. I worked on implementing and comparing three classifiers in **Python and NumPy**, exploring how preprocessing, model parameters, and class imbalance affect their predictions.

[Read the report](report.pdf) · [Explore the models](src/methods) · [View the results](results/evaluation.json)

## What I worked on

- **Implementing the algorithms:** distance-weighted KNN, K-Means with majority-label assignment, and multinomial logistic regression with stable softmax and L2 regularization.
- **Building the data pipeline:** standardization, outlier clipping, scaling, and variance-based feature selection, fitted separately inside each training fold.
- **Comparing models fairly:** nested stratified cross-validation, training-only hyperparameter selection, macro F1, and a majority-class baseline.
- **Making the work reproducible:** fixed seeds, saved predictions and fold indices, dataset and source checksums, automated tests, and GitHub Actions.

The classifiers share a small `fit` / `predict` interface. NumPy handles the numerical work; the model implementations are in [`src/methods`](src/methods).

| Model | Implementation |
| --- | --- |
| KNN | Euclidean distances and inverse-distance votes |
| K-Means | Iterative centroid updates and a learned cluster-to-class mapping |
| Logistic regression | Softmax probabilities and full-batch gradient descent |

## Results

The dataset has **13 features and 5 classes**, with 237 training records and a fixed 60-record test split. Class 4 has only eight training examples, which makes accuracy alone a poor guide to model quality.

The main comparison uses **five outer folds** to estimate performance and **four inner folds** to select parameters. Macro F1 gives each class equal weight. Values below are means ± standard deviations across the outer folds.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Majority baseline | 54.02% ± 0.49 | 0.140 ± 0.001 |
| KNN | 52.37% ± 6.56 | 0.279 ± 0.072 |
| K-Means | 55.68% ± 2.39 | 0.274 ± 0.043 |
| Logistic regression | 56.93% ± 3.77 | 0.292 ± 0.048 |

All three models improve macro F1 over the baseline, but the differences between them are small relative to fold variation. The results highlight the difficulty of learning minority classes from a small dataset.

The fixed test split was used in exploratory parameter searches, so its scores are reported as descriptive results. Parameter selection in `evaluate.py` uses training data only. The report discusses this limitation, numeric encoding of categorical features, and the single initialization used for K-Means.

## Try it

You need **Python 3.10+**. From the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .

# Reproduce the complete model comparison
python3 evaluate.py

# Or evaluate one configuration
python3 main.py --method knn --k 7 --cv-folds 5

# Run the tests
python3 -m unittest discover -s tests -v
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

`evaluate.py` saves the search grids, selected parameters, fold scores, predictions, and provenance in [`results/evaluation.json`](results/evaluation.json). Runs with the same data and software are deterministic; floating-point ties can vary across NumPy versions. `main.py` evaluates a fixed configuration without performing a parameter search.

## Inside the project

| Path | Purpose |
| --- | --- |
| [`src/methods`](src/methods) | NumPy implementations of the three classifiers |
| [`main.py`](main.py) | Evaluate a fixed model configuration |
| [`evaluate.py`](evaluate.py) | Parameter search and nested cross-validation |
| [`tests`](tests) | Checks for predictions, fold coverage, and separation of selection from test data |
| [`experiments`](experiments) | Exploratory parameter sweeps and plotting scripts |
| [`report.tex`](report.tex) | Report source with embedded vector plots |
| [`scripts/sync_report.py`](scripts/sync_report.py) | Generate report tables and plots from the saved results |

After rerunning an evaluation, synchronize the report with:

```sh
python3 scripts/sync_report.py
python3 scripts/sync_report.py --check
```

To build the PDF, use a LaTeX installation containing PGFPlots and run `pdflatex -interaction=nonstopmode -halt-on-error report.tex` twice. For plotting scripts and their optional dependencies, see the [experiments guide](experiments/README.md).

## Team and data

**Omar Berrada · Othmane Housni · Ali El Azdi**

The data comes from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano, shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The course adaptation removes six incomplete records and provides the fixed train/test split. Dataset DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
