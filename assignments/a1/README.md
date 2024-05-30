# A1: Extracting Linguistic Features Using `spaCy`
## Overview

The aim of this project is to extract a variety of linguistic features from a corpus of text using a trained pipeline from ```spaCy```. The specific model used is ```en_core_web_md```, which is a general-purpose English pipeline trained on written text from the internet and consists of a tok2vec, tagger, parser, senter, ner, attribute_ruler, and lemmatizer component.

More specifically, the script extracts the relative frequency of different parts of speech (i.e., nouns, verbs, adjective, and adverbs) per 10,000 words, as well as the total number of unique named entities (i.e., PER = persons, LOC = locations, and ORG = organizations).

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Key Points from the Outputs](#key-points-from-the-outputs)
- [Discussion of Limitations and Possible Improvements](#discussion-of-limitations-and-possible-improvements)

## Repo structure

```bash
a1
├── README.md
├── in
│   ├── a1
│   ├── a2
│   ├── a3
│   ├── a4
│   ├── a5
│   ├── b1
│   ├── b2
│   ├── b3
│   ├── b4
│   ├── b5
│   ├── b6
│   ├── b7
│   ├── b8
│   └── c1
├── out
│   ├── annotations_a1.csv
│   ├── annotations_a2.csv
│   ├── annotations_a3.csv
│   ├── annotations_a4.csv
│   ├── annotations_a5.csv
│   ├── annotations_b1.csv
│   ├── annotations_b2.csv
│   ├── annotations_b3.csv
│   ├── annotations_b4.csv
│   ├── annotations_b5.csv
│   ├── annotations_b6.csv
│   ├── annotations_b7.csv
│   ├── annotations_b8.csv
│   ├── annotations_c1.csv
│   └── feature_extraction_emissions.csv
├── run.sh
└── src
    └── feature_extraction.py
```

## Data Source and Prerequisites

The text corpus used for this project is the *The Uppsala Student English Corpus (USE)* which consists of 1,489 essays in English comprising 1,221,265 words. The essays are <br>
More information about the data, its authors along with a download link to a compressed zip file can be found at the Oxford Text Archive [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457).

Once downloaded and extracted, all folders contained within the dataset should be placed in *./in*. The repo structure should be identical to the one outlined above.

The main script was written and executed using ```Python v.1.89.1```. 
For the processing and analysis of the data, the following packages were used:

```
codecarbon==2.4.2
pandas==2.2.2
spacy==3.7.4
tqdm==4.66.4
```

## Reproducing the Analysis:

To reproduce the analysis, change directory to *a1* and run *run.sh* from the from the terminal:
```bash
cd a1
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
    pip install --upgrade pip
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    ```
4. Runs the main Python script:
    ```
    python src/feature_extraction.py
    ``` 

Once finished, you should now have 14 CSV files in */out* corresponding to the folders in the dataset. 

## Discussion of Limitations and Possible Steps to Improvement

For this project, parts of speech and named entities were extracted from a large corpus of essays written by Swedish university students of English at three different levels. This was done by looping over each txt file within each subfolder and employing an NLP pipeline to to analyze each token within the document. The result were then printed to 14 different CSV files.

Extracting any meaningful information from the resulting data as is would be a tedious and difficult task. As each CSV file contains up to several hundreds rows of data, comparing values and extruding relationships between data points ideally requires further processing of the data such as visualing gathering gathering cumulative sums of all values and visualising key points with graphic charts.

Generally speaking, however, this does not invalidate the method used in this project. Using this approach to NLP, we are able to extract key information which could potentially tell us a great deal about the corpus. In this particular context, extracting linguistic information from student essays across both genres and academic levels might potentially help to shed light on the way students of English develop their linguistic and academic skills. It might tell us how the use of language develops in relation to acadmic progress; For instance, older students might tend to use more descriptive language (i.e. a higher rel. freq. of ADJ and ADV) or more frequently refer to other individuals and associations such as academic theorists and guilds (i.e., a higher rel. freq. of PER and ORG). The same might possibly be the case across different types of assignments and the topics and contents that might be expected from them. 

However, the approach is not without limitations. For one, while the pipeline provided buy `spaCy` is largely both robust and user friendly, it is important to note that no additional adjustment has been done to the model. As a result, the analysis is completely dependant on the quality of the pretrained model which is trained on large amounts of data from the internet (news, blog posts, comments, etc.) and not specifically on academic essays.

Another potential flaw of the analysis lies within the dataset itself. While it is fit for the scope of this assignment, the distribution of essays across university levels is extremely imbalanced with the majority of essays stemming from first semester students while only seven essays were written by students on their third semester. As such, it would be impossible to actually provide any meaningful conclusions in regard to comparing the use of language across semesters. 