import os
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from joblib import dump, load
import utils.classifier_utils as clf
from codecarbon import EmissionsTracker

def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)

def preprocess_data(data):
    X = data["text"]
    y = data["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 lowercase=True,
                                 max_df=0.95,
                                 min_df=0.05,
                                 max_features=500)
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()
    return X_train_feats, X_test_feats, feature_names, vectorizer

def train_classifier(X_train_feats, y_train):
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    return classifier

def evaluate_classifier(classifier, X_test_feats, y_test, vectorizer):
    y_pred = classifier.predict(X_test_feats)
    clf.show_features(vectorizer, y_test, classifier, n=20)
    metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test_feats, y_test, cmap=plt.cm.Blues, labels=["FAKE", "REAL"])
    classifier_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    return classifier_metrics

def save_model(classifier, vectorizer):
    dump(classifier, "models/LR_classifier.joblib")
    dump(vectorizer, "models/LR_tfidf_vectorizer.joblib")

def plot_learning_curve(X, y):
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = LogisticRegression(random_state=42)
    clf.plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

def train_lrc(input_path, output_path):
    data = load_data(input_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_feats, X_test_feats, _, vectorizer = vectorize_text(X_train, X_test)
    classifier = train_classifier(X_train_feats, y_train)
    classifier_metrics = evaluate_classifier(classifier, X_test_feats, y_test, vectorizer)
    df = pd.DataFrame(classifier_metrics).transpose()
    df.to_csv(os.path.join(output_path, "lrc_report.csv"))
    plot_learning_curve(vectorizer.fit_transform(data["text"]), data["label"])
    save_model(classifier, vectorizer)

def eval_sentence(sentence):
    loaded_classifier = load("models/LR_classifier.joblib")
    loaded_vect = load("models/LR_tfidf_vectorizer.joblib")
    test_sentence = loaded_vect.transform([sentence])
    print(f'The sentece you typed was classified as {loaded_classifier.predict(test_sentence)}')

def main(sentence):
    input_path = os.path.join("in", "fake_or_real_news.csv")
    output_path = "out"

    tracker = EmissionsTracker(project_name="Fake news classification",
                                experiment_id="logistic_regression",
                                output_dir=output_path,
                                output_file="logistic_regression_emissions.csv")

    tracker.start_task("Entire script")

    if os.path.exists(os.path.join("models", "LR_classifier.joblib")):
        eval_sentence(sentence)
    else:
        train_lrc(input_path, output_path)
        eval_sentence(sentence)

    tracker.stop()
    print(f'Saved emissions report to {os.path.join(output_path, "logistic_regression_emissions.csv")}')
if __name__ == "__main__":
    main()