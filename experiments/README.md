# Experiment notebook, without the notebook

These scripts contain the parameter searches and plots from the course project. Run them from this folder so the relative data paths work:

```sh
python3 -m pip install -e '..[experiments]'
python3 plot_logistic_params.py --help
python3 find_best_params.py --help
python3 main.py --method knn --data_path features.npz --K 7 --test
```

The plotting scripts may open a window and write figures to the current folder. For a terminal without a display, use `MPLBACKEND=Agg`.

The `src` folder here belongs to these experiments; the main evaluation code is one level up. The report records the course runs, while the main entry point uses a fixed seed and fits preprocessing separately in each cross-validation fold.

The image-loading and neural-network templates in `src` are separate from the heart disease experiments. They need OpenCV or PyTorch if used; neither is needed for the three classifiers documented in the main README.
