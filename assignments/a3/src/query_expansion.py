import os
import re
import pandas as pd
import gensim
import gensim.downloader as api
import argparse
from codecarbon import EmissionsTracker

def load_model():
    print("Fetching model...")
    model = api.load("glove-wiki-gigaword-50")
    return model

def initialize_data():
    filepath = os.path.join(
        "in",
        "Spotify Million Song Dataset_exported.csv"
    )
    data = pd.read_csv(filepath)
    return data

def clean_lyrics_data(data):
    print("Cleaning lyrics...")
    # Clean lyrics
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # remove punctuation
    data['text'] = data['text'].apply(lambda x: re.sub(r'\d+', '', x))  # remove numbers
    data['text'] = data['text'].apply(lambda x: re.sub(r'\s+',' ', x))  # remove extra whitespace
    data['text'] = data['text'].apply(lambda x: x.lower())  # convert to lowercase
    return data

def expand_query(word, model):
    #Expand query with similar words, exlude similarity score
    print(f"Expanding query for word: {word}")
    similar_words = [w for w, _ in model.most_similar(word, topn=10)]
    print(f"Similar words: {similar_words}")
    similar_words.extend([word])
    return similar_words

def filter_by_artist(artist, data):
    artist_lower = artist.lower()
    data['artist'] = data['artist'].str.lower()
    artist_songs = data[data["artist"] == artist_lower]
    return artist_songs

def calculate_percentage(artist, artist_songs, similar_words, word):
    print(f"Artist songs: {artist_songs.shape[0]}")
    matching_songs = artist_songs
    for w in similar_words:
        matching_songs = matching_songs[matching_songs['text'].str.contains(word, case=False)]
    percentage = (len(matching_songs) / len(artist_songs)) * 100
    print(f'{percentage:.2f}% of {artist}\'s songs contain the word "{word}" or similar')

def get_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word','-w',
                        help = "Search word used for query expansion",
                        required = True
    )
    parser.add_argument('--artist','-a',
                        help = "Artist to filter songs by",
                        required = True
    )
    return parser.parse_args()

def main():
    output_path = "out"
    tracker = EmissionsTracker(project_name="Query expansion",
                                experiment_id="query_expansion",
                                output_dir=output_path,
                                output_file="emissions.csv",
                                measure_power_secs=5,
                                log_level = "error")

    args = get_program_args()
    print(f"Word: {args.word}, Artist: {args.artist}")

    task_name = "load model"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    model = load_model()
    load_model_emissions = tracker.stop_task()

    task_name = "read and preprocess data"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    data = initialize_data()
    data = clean_lyrics_data(data)
    process_data_emissions = tracker.stop_task()

    task_name = "apply model"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    similar_words = expand_query(args.word, model)
    model_emissions = tracker.stop_task()

    task_name = "calculate percentage"
    print(f"Starting task: {task_name}")
    tracker.start_task(task_name)
    artist_songs = filter_by_artist(args.artist, data)
    calculate_percentage(args.artist, artist_songs, similar_words, args.word)
    percentage_emissions = tracker.stop_task()

    tracker.stop()
if __name__== "__main__":
    main()