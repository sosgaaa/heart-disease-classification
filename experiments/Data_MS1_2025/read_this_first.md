# Processed heart disease data

This course adaptation of the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) uses the processed Cleveland records. Six incomplete records were removed from the original 303 observations, leaving **297 records with 13 features**.

The supplied fixed split contains **237 training records and 60 test records**. The NPZ arrays are named `xtrain`, `xtest`, `ytrain`, and `ytest`. Labels are stored as floating-point values representing integer classes; the evaluation loader converts them to integers.

## Features

The [UCI documentation](https://archive.ics.uci.edu/dataset/45/heart+disease) describes the feature meanings and category values.

| Feature | Type |
| --- | --- |
| age | Integer |
| sex | Categorical |
| cp | Categorical |
| trestbps | Integer |
| chol | Integer |
| fbs | Categorical |
| restecg | Categorical |
| thalach | Integer |
| exang | Categorical |
| oldpeak | Real |
| slope | Categorical |
| ca | Integer |
| thal | Categorical |

## Class distribution

There are five classes, labelled 0 to 4. Label 0 denotes absence of disease.

| Class | Training records | Test records |
| --- | ---: | ---: |
| 0 | 128 | 32 |
| 1 | 41 | 13 |
| 2 | 30 | 5 |
| 3 | 30 | 5 |
| 4 | 8 | 5 |

The imbalance motivates reporting macro F1 and per-class scores alongside accuracy. Attribution and the CC BY 4.0 dataset license are documented in the project README.
