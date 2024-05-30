import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from codecarbon import EmissionsTracker

def load_model():
    print("Loading classifier...")
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        top_k=None)
    return classifier

def initialize_data(path_in):
    data = pd.read_csv(path_in)
    data['Season'] = data['Season'].str.extract(r'(\d+)', expand=False).astype(int)
    data['Episode'] = data['Episode'].str.extract(r'(\d+)', expand=False).astype(int)
    return data

def predict_emotions(data, classifier):
    predictions = []
    print("Analysing sentences...")
    for sentence in tqdm(data["Sentence"]):
        try:
            predictions.append(classifier(str(sentence), top_k=1))
        except ValueError:
            print(f'ValueError; Skipping {sentence}')
            continue
    return predictions

def append_labels(predictions, data):
    labels = []
    label_scores = []
    print("Appending labels...")
    for result in tqdm(predictions):
        labels.append(result[0]['label'])
        label_scores.append(result[0]['score'])
    data['Label'] = labels
    data['Score'] = label_scores
    unique_labels = data['Label'].unique()
    return unique_labels

'''
def plot_distribution(data, labels):
    # Plot the distribution for each season
    seasons = sorted(data['Season'].unique())
    max_frequency = 0  # Initialize maximum frequency
    for season in seasons:
        season_data = data[data['Season'] == season]
        label_counts = season_data['Label'].value_counts().reindex(labels, fill_value=0)
        max_frequency = max(max_frequency, label_counts.max())  # Update maximum frequency
        label_counts.plot(kind='line', marker='o', figsize=(10, 6))
        plt.title(f'Label distribution for Season {season}')
        plt.xlabel('Label')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.ylim(0, max_frequency)  # Set y-axis limit to the maximum frequency
        plt.grid(True)
        plt.show()
'''

def plot_frequency_distribution(data, labels, path_out):
    label_counts_by_season = {season: {} for season in data['Season'].unique()}

    # Accumulate label counts for each season
    for season, season_data in data.groupby('Season'):
        label_counts = season_data['Label'].value_counts().reindex(labels, fill_value=0)
        label_counts_by_season[season] = label_counts

    # Convert dictionary to DataFrame for easier plotting
    label_counts_df = pd.DataFrame(label_counts_by_season, index=labels)

    # Plot the grouped bar plot
    label_counts_df.plot(kind='bar', figsize=(12, 8))

    plt.title('Distribution of labels for each Season')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.legend(title='Season')
    plt.grid(True)
    plt.savefig(os.path.join(path_out, "Frequency distribution.png"))
    plt.show()
    print(f'Plot saved as "Frequency distribution.png" to {path_out}')

def plot_relative_freq(data, labels,path_out):
    # Initialize an empty dictionary to store relative label frequencies for each season
    relative_frequencies_by_season = {}
    for season in data['Season'].unique():
        relative_frequencies_by_season[season] = []

    # Calculate relative frequencies for each season
    for season, season_data in data.groupby('Season'):
        label_counts = season_data['Label'].value_counts().reindex(labels, fill_value=0)
        total_labels = label_counts.sum()
        relative_frequencies = label_counts / total_labels
        relative_frequencies_by_season[season] = relative_frequencies

    # Convert dictionary to DataFrame for easier plotting
    relative_frequencies_df = pd.DataFrame(relative_frequencies_by_season, index=labels)

    # Plot the grouped bar plot
    relative_frequencies_df.plot(kind='bar', figsize=(12, 8))

    plt.title('Relative frequency distribution of labels across Seasons')
    plt.xlabel('Label')
    plt.ylabel('Relative Frequency')
    plt.xticks(rotation=45)
    plt.legend(title='Season')
    plt.grid(True)
    plt.savefig(os.path.join(path_out, "Relative frequency.png"))
    plt.show()
    print(f'Plot saved as "Relative frequency.png" to {path_out}')

def main():
    path_in = os.path.join("in","Game_of_Thrones_Script.csv")
    path_out = os.path.join("out")

    tracker = EmissionsTracker(project_name="Emotion analysis",
                            experiment_id="emotion_analysis",
                            output_dir=path_out,
                            output_file="emissions.csv",
                            measure_power_secs=5,
                            log_level = "error")

    task_name = "load model"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    classifier = load_model()
    load_model_emissions = tracker.stop_task()

    task_name = "initialize data"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    data = initialize_data(path_in)
    init_data_emissions = tracker.stop_task()

    task_name = "predict labels"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    predictions = predict_emotions(data, classifier)
    prediction_emissions = tracker.stop_task()

    task_name = "append labels"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    labels = append_labels(predictions, data)
    append_labels_emissions = tracker.stop_task()

    task_name = "plot data"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    plot_frequency_distribution(data, labels, path_out)
    plot_relative_freq(data, labels, path_out)
    plot_function_emissions = tracker.stop_task()

    tracker.stop()
    
if __name__ == "__main__":
    main()