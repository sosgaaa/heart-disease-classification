# Heart Disease Classification 🧠

Three models, one dataset, and quite a few plots.

I worked on this project at EPFL for **Introduction to Machine Learning (CS-233)** with **Othmane Housni and Ali El Azdi**. We compared three methods on a heart disease dataset and looked at how preprocessing and parameter choices changed the results.

The models are implemented in **Python with NumPy**. Working through the algorithms made it easier to see what each one was actually doing, rather than just calling a library and reading a score.

## The three methods

| Model | The idea |
| --- | --- |
| K-nearest neighbours | Look at nearby samples and let them vote |
| K-means | Group similar samples, then give each group its most common label |
| Logistic regression | Learn weights and use softmax to predict a class |

The processed dataset has **297 samples**, **13 features**, and **5 classes**. It includes a fixed split of 237 training samples and 60 test samples.

## Run an evaluation

You need **Python 3.10+**. From the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .

python3 main.py --method knn --cv-folds 5
python3 main.py --method kmeans --cv-folds 5
python3 main.py --method logistic_regression --cv-folds 5
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Each command prints cross-validation results and the final test accuracy and macro F1 score. Scaling and feature selection are fitted on the training data of each fold.

You can also change the parameters:

```sh
python3 main.py --method knn --k 5 --cv-folds 5
python3 main.py --method logistic_regression --lr 0.1 --max-iters 500
python3 -m unittest discover -s tests -v
```

## Report and experiments

The [project report](report.pdf) contains our course experiments, figures, and discussion. The main lesson was that preparing the data and choosing a fair evaluation can matter as much as choosing the model.

| Path | What's inside |
| --- | --- |
| `main.py` | Evaluation entry point |
| `src/methods` | The three models |
| `features.npz` | Processed data and train/test split |
| `screens` | Figures used in the report |
| `experiments` | Parameter searches and plotting scripts |
| `tests` | Checks for the evaluation pipeline |

For the plotting scripts, install the extra dependencies and run them from `experiments`:

```sh
python3 -m pip install -e '.[experiments]'
cd experiments
python3 plot_logistic_params.py --help
```

## Team and data

Omar Berrada, Othmane Housni, and Ali El Azdi.

The data comes from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano, shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The course version removes six incomplete samples and provides the fixed train/test split. Dataset DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
