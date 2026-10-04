# Heart disease classification

Introduction to Machine Learning course project comparing k-nearest neighbours, k-means with majority-label assignment, and multinomial logistic regression on a 297-row, 13-feature heart disease dataset. The report and its figures are included.

## Course and starting point

[Introduction to machine learning (CS-233)](https://edu.epfl.ch/studyplan/en/bachelor/computer-science/coursebook/introduction-to-machine-learning-CS-233) introduces learning models and their rigorous evaluation on data. This project implements and compares classification and clustering methods in Python, starting from a course method template and the processed feature dataset in `features.npz`.

## Run

Requires Python 3.10+ and NumPy.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
python3 main.py --method knn --cv-folds 5
python3 main.py --method kmeans --cv-folds 5
python3 main.py --method logistic_regression --cv-folds 5
python3 -m unittest discover -s tests
```

`features.npz` contains the train/test split. The script fits scaling and feature selection on the training portion of each fold, then evaluates the configured models on the held-out test set. The report documents the course experiments.

## Project team and data

Project team: Othmane Housni, Omar Berrada, and Ali El Azdi. The processed features are stored in `features.npz`.
