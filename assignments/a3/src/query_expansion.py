import os
import re
import spacy
import pandas as pd
import gensim
import gensim.downloader as api
import argparse

model = api.load("glove-wiki-gigaword-50")

word = "her"
artist = "Abba"

def initialize_data():
    filepath = os.path.join(
        "..",
        "in",
        "Spotify Million Song Dataset_exported.csv"
    )
    return pd.read_csv(filepath)


def clean_lyrics_data(data):
# Clean lyrics
    data["text"] = data["text"].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # remove punctuation
    data["text"] = data["text"].apply(lambda x: re.sub(r'\d+', '', x))  # remove numbers
    data["text"] = data["text"].apply(lambda x: re.sub(r'\s+',' ', x))  # remove extra whitespace
    data["text"] = data["text"].apply(lambda x: x.lower())  # convert to lowercase

    return data["text"]



def expand_query(word):
    #Expand query with similar words, exlude similarity score
    similar_words = [w for w, _ in model.most_similar(word, topn=10)]

    return similar_words.extend([word])


def filter_by_artist(artist):

    artist_lower = artist.lower()

    data['artist'] = data['artist'].str.lower()

    return data[data["artist"] == artist_lower]


def calculate_percentage(artist_songs):

    matching_songs = artist_songs

    for w in similar_words:
        matching_songs = matching_songs[matching_songs['text'].str.contains(word, case=False)]

    percentage = (len(matching_songs) / len(artist_songs)) * 100
    print(f'{percentage}% of {artist}\'s songs contain the word "{word}" or similar')


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
    get_program_args()

if __name__== "__main__":
    args = get_program_args()