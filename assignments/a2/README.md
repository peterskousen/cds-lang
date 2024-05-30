# A2: Text Classification Benchmarks
## Overview
In this project I aim to train two binary classification models to predict whether or not a user-defined input string should be classified as *fake news* or *real news*. For this, two basic machine learning algorithms are tested against each other, namely a logistic regression model and a multi-layered perceptron. These are both supervised learning algorithms that utilize labeled datasets to predict outcomes. The models are employed using `scikit-learn`'s built-in `LogisticRegression()` and `MLPClassifier()`. To optimize the process of rerunning the script multiple times, the scripts checks the `models` directory for any existing trained models. Depending on this, the program either trains the given model or skips the process entirely in favor of loading in an exising model file. <br>

Before training the respective classifiers, the dataset is split into two subsets for training and testing (80/20 distribution) using `train_test_split`. Then, using `TfidfVectorizer()`, the text data is turned into numerical representations using TF-IDF vectorization with a upper and lower term frequency threshold of 95% and 5% respectively to ensure that the most common and most rare words do not negatively affect the performance of the model. The parameters used are identical between the two models. <br>
The MLP classifier is configured with a single hidden layer of 20 neurons. It uses a sigmoid function for activation and iterates over the training up to 1000 times before stopping.

After training, the models are evaluated with a confusion matrix, a loss curve is plotted for the MLP classifier, and the trained models are stored for easy future retrieval. 

The `src` folder contains three scripts: `logistic_regression.py`, `mlp_classifier.py`, and `main.py`. The latter script accepts two arguments and executes either `logistic_regression.py` or `mlp_classifier.py` with an input string. For more information about this, please refer to the section [Reproducing the Analysis](#reproducing-the-analysis).

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Results](#results)

## Repo structure

```bash
.
├── README.md
├── in
│   └── fake_or_real_news.csv
├── models
│   ├── LR_classifier.joblib
│   ├── LR_tfidf_vectorizer.joblib
│   ├── MLP_classifier.joblib
│   └── MLP_tfidf_vectorizer.joblib
├── out
│   ├── emissions_base_0b43edb5-6759-4615-8d5b-0696d3b07353.csv
│   ├── emissions_base_f905d2d2-44ef-41da-99a8-ffd038352607.csv
│   ├── logistic_regression_emissions.csv
│   ├── lrc_report.csv
│   ├── mlp_emissions.csv
│   └── mlp_report.csv
├── requirements.txt
├── setup.sh
└── src
    ├── logistic_regression.py
    ├── main.py
    ├── mlp_classifier.py
    └── utils
        └── classifier_utils.py
```

## Data Source and Prerequisites

The dataset used for this project is a collection of 10,557 news stories, 50% of which are labelled "TRUE", the other 50% labelled "FALSE". Aditional information along with a download link to the CSV file can be found on Kaggle via [this link](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news). The CSV file should be placed directly in the `in` folder as demonstrated in the repo structure above. 

The main script was written and executed using ```Python v.1.89.1```. 
For the processing and analysis of the data, the following packages were used:

```
codecarbon==2.4.2
joblib==1.4.2
matplotlib==3.9.0
numpy==1.26.4
pandas==2.2.2
scikit_learn==1.5.0
seaborn==0.13.2
```

## Reproducing the Analysis:

To reproduce the analysis, firstly change directory to `a2` and run `setup.sh` from the from the terminal:
```bash
cd local_path/to/a2
bash setup.sh
``` 
`setup.sh` performs the following actions:
1. Sets up a virtual environment called *.venv* in the root directory using the `venv` module:
    ```sh
    python -m venv .venv
    ```
2. Activates the environment:
    ```sh
    source .venv/bin/activate
    ```
3. Fetches and installs required dependencies:
    ```
    pip install -r requirements.txt
    ``` 
4. Deactivates the environment:
    ```
    deactivate
    ``` 
<br>
After the directory is set up, you will have to manually activate the virtual environment again:

```bash
source .venv/bin/activate
```

Now you can run the main program used for conducting the classification. Two arguments are required to run the program: `-i | --input_sentence`, which is the string that will be classified as either real or fake news, and `-c | --classification_type` which can be set to either `mlp` or `lr` to determine the use of a neural network or logistic regression model, respetively. For instance:

```
python src/main.py -i "Hillary is a reptillian" -c mlp
```
This results in a terminal output of either `[REAL]` for `[FAKE]` along with a classification report in the `out` folder.

## Results:

### Logistic Regression

| Class        | Precision | Recall   | F1-Score | Support |
|--------------|-----------|----------|----------|---------|
| FAKE         | 0.8930    | 0.8774   | 0.8851   | 628     |
| REAL         | 0.8815    | 0.8967   | 0.8891   | 639     |
| **Accuracy** | **0.8871** | **0.8871** | **0.8871** | **0.8871** |
| **Macro avg**| 0.8873    | 0.8871   | 0.8871   | 1267    |
| **Weighted avg** | 0.8872 | 0.8871   | 0.8871   | 1267    |

### Neural Network

| Class        | Precision | Recall   | F1-Score | Support |
|--------------|-----------|----------|----------|---------|
| FAKE         | 0.8992    | 0.8662   | 0.8824   | 628     |
| REAL         | 0.8731    | 0.9045   | 0.8885   | 639     |
| **Accuracy** | **0.8856** | **0.8856** | **0.8856** | **0.8856** |
| **Macro avg**| 0.8861    | 0.8854   | 0.8855   | 1267    |
| **Weighted avg** | 0.8860 | 0.8856   | 0.8855   | 1267    |

<br>

Comparing the classifications reports for the two classifiers it becomes evident that they both perform reasonably well at the given task. Both of them exhibit fairly high precision and recall with scores in the high 80s across the board. The overall accuracy is ever so slightly higher for the logistic regression classifier, which reaches an accuracy score of 0.887. In contrast, the MLP classifier achieves an accuracy score of 0.885. The difference between these scores is extremely small and would be negligible in any real-world scenario.