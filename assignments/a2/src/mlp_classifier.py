import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import utils.classifier_utils as clf
import matplotlib.pyplot as plt

def train_mlp():
    filepath = os.path.join(
        "..",
        "..",
        "..",
        "data",
        "fake_or_real_news.csv"
    )
    data_csv = pd.read_csv(filepath, index_col=0)

    X = data_csv["text"]
    y = data_csv["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    vectorizer = TfidfVectorizer(ngram_range = (1,2),
                                 lowercase =  True,
                                 max_df = 0.95,
                                 min_df = 0.05,
                                 max_features = 500)

    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()

    classifier = MLPClassifier(activation = "logistic", 
                               hidden_layer_sizes = (20,), max_iter=1000,
                               random_state = 42)

    classifier.fit(X_train_feats, y_train)

    y_pred = classifier.predict(X_test_feats)
    print(y_pred[:20])

    from joblib import dump, load
    dump(classifier, "../models/MLP_classifier.joblib")
    dump(vectorizer, "../models/MLP_tfidf_vectorizer.joblib")

    #Evaluate model
    metrics.ConfusionMatrixDisplay.from_estimator(classifier,
                                                  X_train_feats,
                                                  y_train,
                                                  cmap=plt.cm.Blues,
                                                  labels=["FAKE", "REAL"])

    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)
    
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.show()

def eval_sentence(sentence):
    from joblib import dump, load
    loaded_clf = load("../models/MLP_classifier.joblib")
    loaded_vect = load("../models/MLP_tfidf_vectorizer.joblib")
    test_sentence = loaded_vect.transform([sentence])
    print(loaded_clf.predict(test_sentence))