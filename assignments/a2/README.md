# A2: Text Classification Benchmarks
## Overview

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Key Points from the Outputs](#key-points-from-the-outputs)
- [Discussion of Limitations and Possible Improvements](#discussion-of-limitations-and-possible-improvements)

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

To reproduce the analysis, change directory to *a3* and run *setup.sh* from the from the terminal:
```bash
cd a3
bash setup.sh
``` 
*run.sh* performs the following actions:
1. Sets up a virtual environment called *.venv* in the root directory using ```venv```:
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

## Key Points from the Outputs:

## Discussion of Limitations and Possible Steps to Improvement