import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import nn_classifier as nn
import lrc_classifier as lrc
import argparse

def get_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_type','-cl',
                        required = True,
                        help = "Type of classifier to execute"
    )
    parser.add_argument('--input_sentence','-i',
                        help = "Input sentence to analyse",
                        choices = ["nn", "lrc"],
                        required = True,
                        type = str
    )
    return parser.parse_args()

def main():
    get_program_args()

if __name__=="__main__":
    args = get_program_args()
    if args.classification_type == "nn":
        nn.eval_sentence(args.input_sentence)
    elif args.classification_type == "lrc":
        lrc.eval_sentence(args.input_sentence)