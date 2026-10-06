# Experiments

Parameter sweeps and plotting scripts for exploring preprocessing, feature counts, validation splits, and model behaviour. Figures from these experiments are stored in `../screens`.

For the model comparison presented in the report, run `python3 evaluate.py` from the repository root. It fits preprocessing inside each fold and selects parameters using training data only.

## Run a script

Install the plotting dependencies from the repository root, then switch to this folder:

```sh
python3 -m pip install -e '.[experiments]'
cd experiments
python3 plot_logistic_params.py --help
python3 find_best_params.py --help
```

Scripts use relative data paths, so run them from `experiments`. Plots may open a window and save images in the current folder. On a machine without a display, set `MPLBACKEND=Agg`.

To reproduce the fixed configurations discussed in the report:

```sh
python3 main.py --method knn --K 7 --test
python3 main.py --method kmeans --K 34 --test
python3 main.py --method logistic_regression --lr 0.1 --max_iters 500 --test
```

## How to interpret these experiments

These are exploratory runs, with different preprocessing and configurations from the nested-CV comparison. Some scripts fit transforms before splitting or select validation proportions by their scores. The hold-out results for KNN and logistic regression use 15% validation; K-Means uses 22%.

`plot_logistic_params.py` searches using fixed-test accuracy. `plot_logistic_combined.py` selects by the average of CV and fixed-test accuracy. Their scores describe parameter sensitivity on inspected data and should not be interpreted as an independent final evaluation. Use `../evaluate.py` for training-only selection and nested validation.

The local `src` package belongs to these experiments. Its image-loading utilities require OpenCV when used with image datasets; OpenCV is not needed for the three NumPy classifiers.
