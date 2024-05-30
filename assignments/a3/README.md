# A3: Query Expansion with Word Embeddings
## Overview

The goal of this assignment is to exploit word embeddings to create query expansions, i.e., finding the 10 most similiar words to the initial user input and including those in a search query. This will be done on the basis of a dataset consisting of lyrics from 57,650 English-language songs. With two arguments, the user is able to select a specific artist and word and receive a value denoting how many of that artist's songs contain both the specific word and related words.

To do this I firstly utilize `gensim.downloader.load()` to download and load a pretrained model with 400k vectors. Once the model is loaded, the program preprocesses the text corpus by removing redundant characters such as punctuation, numbers, and extra whitespace. <br> Aditionally, all text is converted to lowercase. <br> The built-in `gensim` function `most_similar()` then computes the cosine similarity between vectors of the given word along with the vectors of all words in the model and returns the 10 closest. <br> Finally, the percentage of the given artist's song contained the queried word is calculated and returned to the user as a string in the CLI. 

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Results](#results)

## Repo structure

```bash
a3
├── README.md
├── in
│   └── Spotify Million Song Dataset_exported.csv
├── out
│   └── a3_query_expansions_emissions.csv
├── requirements.txt
├── setup.sh
└── src
    └── query_expansion.py
```

## Data Source and Prerequisites
*In this assignment, we're going to be working with a corpus of lyrics from 57,650 English-language songs. You can find a link to the dataset [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs).*

from the `GloVe` project (https://nlp.stanford.edu/projects/glove/)

The main script was written and executed using ```Python v.1.89.1```. 
For the processing and analysis of the data, the following packages were used:

```
codecarbon==2.4.2
gensim==4.3.2
pandas==2.2.2
scipy==1.11
```

## Reproducing the Analysis:

To reproduce the analysis, change directory to *a3* and run *setup.sh* from the from the terminal:
```bash
cd a3
bash setup.sh
``` 
*setup.sh* performs the following actions:
1. Sets up a virtual environment called ``.venv`` in the root directory using the ``venv`` module:
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

## Results

```
Word: night, Artist: pink floyd
Fetching model...
Cleaning lyrics...
Expanding query for word: night
Similar words: ['day', 'weekend', 'morning', 'days', 'evening', 'saturday', 'sunday', 'afternoon', 'went', 'hours']
Artist songs: 111
```

The resulting data acquired from running the program is presented as a string output in the CLI. Thus, no data is stored as a result of executing from the script. <br>
In terms of the approach applied in this project, however, one drawback is that the input query has to be quite precise. For instance, if one wanted to look for a specific word in The Beatles' catalog of songs, it is not enough to merely input "beatles" as the query. For the script to correctly identify the artist, it has to match the string in the database precisely, including spaces and definite articles. To optimize the functionality of the program, future versions of the script could incorporate a fuzzy string matching algorithm which identifies strings in the dataset that are similar to the query.