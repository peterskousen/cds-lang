# A4: Emotion Analysis with Pretrained Language Models
## Overview

The objective of this project is to analyze the emotional profile of *Game of Thrones* using pretrained language models. By examining the scripts from all seasons, I aim to uncover how the emotional tone of the show evolves over time. This analysis leverages the pretrained BERT-style model ``emotion-english-distilroberta-base`` from ``HuggingFace`` which classifies emotions in English text data, including *anger*, *disgust*, *fear*, *joy*, *neutral*, *sadness*, and *surprise*. The model was trained on subsets from 6 different English datasets including [texts from Twitter, Reddit, student self-reports, and utterances from TV dialogues](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base#:~:text=texts%20from%20Twitter%2C%20Reddit%2C%20student%20self%2Dreports%2C%20and%20utterances%20from%20TV%20dialogues).

More specifically, the program is written to predict emotion scores for each of the show's 23,911 lines of dialogue, plotting the distribution of emotions for each season, and analyzing the relative frequency of each emotion across all seasons. 

## Table of Contents

- [Repo Structure](#repo-structure)
- [Data Source and Prerequisites](#data-source-and-prerequisites)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Key Points from the Outputs](#key-points-from-the-outputs)
- [Discussion of Limitations and Possible Improvements](#discussion-of-limitations-and-possible-improvements)

## Repo structure

```bash
a4
├── in
│   └── Game_of_Thrones_Script.csv
├── out
│   ├── Frequency distribution.png
│   ├── Relative frequency.png
│   ├── a4_emotion_analysis_emissions.csv
│   └── emissions_base_0c160135-cfd9-42f0-8345-3aab7b8d3748.csv
├── requirements.txt
├── run.sh
└── src
    └── emotion_analysis.py
```

## Data Source and Prerequisites
The text corpus used for this project is the entire script of the *Game of Thrones* TV series. This comprises 23,911 lines of dialogue spanning eight seasons and 73 episodes. More information about the dataset and its author along with a download link to a compressed CSV file can be found on [Kaggle](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select=Game_of_Thrones_Script.csv).

The main script was written and executed using ```Python v.1.89.1```. 
For the processing and analysis of the data, the following packages were used:

```
codecarbon==2.4.2
matplotlib==3.9.0
numpy==1.26.4
pandas==2.2.2
tqdm==4.66.4
transformers==4.41.1
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

<div style="display: flex;">
    <img src="./out/Frequency%20distribution.png" alt="Frequency Distribution" style="width: 45%; margin-right: 5px;">
    <img src="./out/Relative%20frequency.png" alt="Relative Frequency" style="width: 45%; margin-left: 5px;">
</div>

*For larger, hi-res versions of the images, please see `out`*

<br>

The analysis reveals several key insights about the emotional landscape of Game of Thrones. Firstly, though, it might be worth noting that although the *neutral* class is far more prevalent than any other label, it is not orignally part of Ekman's 6 basic emotions upon which the classification is based. Though it does serve as a substitue for any sort of emotion that does not fall within the threshold other 6, it arguably does not provide much meaningful information about the emotional distribution or development of the story. Therefore, it will be regarded slighty less in the following interpretiations.

Looking at the frequency distribution of the labels for each seasons, it appears that for several of the labels, season 2 quite clearly matches or outnumbers other seasons. This is particularly salient for the *neutral*, *anger*, and *fear* labels. This observations becomes particularly interesting in light of the fact that at least among the first six seasons, [season 2 has the shortest runtime](https://www.statista.com/statistics/1011409/game-of-thrones-seasons-length/) of the bunch, possibly indicating that this season might be slightly more dialog-driven compared to others. 

While the two bar plots display quite similar trends in the data, the relative frequency distribution provides a very clear picture of a distinct pattern that emerges across the seasons, namely that certain emotions are far more notable throughout the entire series. Disregarding `neutral`, the most common emotions exhibited in all 8 seasons are *anger*, *surprise*, and *disgust*, while *sadness*, *fear*, and *joy* are all much less prominent. This could be interpreted as reflecting the politically polarized and culturally diverse nature of the *Game of Thrones* universe and the feelings the inhabitants exhibit towards each other.

| **Label** | **Season w/ highest rel. freq.** |
|:---------:|:---------------------------:|
| Disgust   | Season 3                    |
| Surprise  | Season 1                    |
| (Neutral) | (Season 7)                  |
| Fear      | Season 7                    |
| Anger     | Season 8                    |
| Joy       | Season 5                    |
| Sadness   | Season 6                    |

## Discussion of Limitations and Possible Steps to Improvement

While the analysis provides valuable insights, it is not without limitations. Foe example, the pretrained model used is trained on general English text from the internet, which may not perfectly capture the unique language and emotional nuances of the *Game of Thrones* scripts. This could potentially explain the high frequency of *neutral* labels. Further, the quality and consistency of the script data can affect the accuracy of emotion predictions. Any errors or inconsistencies in the data might lead to misleading results. Overall however, for the purposes of this particular analysis, the chosen model and approach appears to produce some quite satisfactory data.