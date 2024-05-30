# A4: Emotion Analysis with Pretrained Language Models
## Overview

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Key Points from the Outputs](#key-points-from-the-outputs)
- [Discussion of Limitations and Possible Improvements](#discussion-of-limitations-and-possible-improvements)

## Repo structure

```bash

```

## Data Source and Prerequisites


The main script was written and executed using ```Python v.1.89.1```. 
For the processing and analysis of the data, the following packages were used:

```

```

## Reproducing the Analysis:

To reproduce the analysis, change directory to *a4* and run *run.sh* from the from the terminal:
```bash
cd a4
bash run.sh
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
4. Runs the main Python script:
    ```
    python src/emotion_analysis.py
    ``` 

## Key Points from the Outputs:

## Discussion of Limitations and Possible Steps to Improvement