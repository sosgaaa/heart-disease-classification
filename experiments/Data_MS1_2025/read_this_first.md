# Data Description for Project Milestone 1

This file aims to provide some information about the processed data itself. It may be useful for description writing and implementing baseline methods. Feel free to delete this file when finally published to moodle.

The original dataset is located at [Here](https://archive.ics.uci.edu/dataset/45/heart+disease), contains 13 features and 303 samples.

### Processing Steps
- As the original data has 6 samples that has some fields missing. I manually remove them. So the processed dataset has totally 297 samples.
- Then I randomly split them into training/test sets with 237/60 samples.

### Dataset Statistic
- It is a 5-class classification problem, with class number from 0 to 4. Note that for the label, I currently save them as **float**. This may cause problem when using cross-entropy loss from torch. (but I think torch is not used in MS1?)
- For the 13 features, their original meaning could be found at [Here](https://archive.ics.uci.edu/dataset/45/heart+disease). I list their type below. For the categorial features, their possible choices could also be found on the website above.

| Feature     |    Type     |
| ----------- | ----------- |
| age         | integer     |
| sex         | categorial  |
| cp          | categorial  |
| trestbps    | integer     |
| chol        | integer     |
| fbs         | categorial  |
| restecg     | categorial  |
| thalach     | integer     |
| exang       | categorial  |
| oldpeak     | integer     |
| slope       | categorial  |
| ca          | integer     |
| thal        | categorial  |

- the number of each class (0-4) are
    - training set: [128, 41, 30, 30, 8]
    - test set: [32, 13, 5, 5, 5] 

    so we may need to remind students to think about the imbalanced problem.