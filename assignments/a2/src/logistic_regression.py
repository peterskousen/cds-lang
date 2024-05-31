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

def load_data(filepath, tracker):
    tracker.start_task("Load Data")
    data = pd.read_csv(filepath, index_col=0)
    tracker.stop_task("Load Data")
    return data

def preprocess_data(data, tracker):
    tracker.start_task("Preprocess Data")
    X = data["text"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tracker.stop_task("Preprocess Data")
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test, tracker):
    tracker.start_task("Vectorize Text")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 lowercase=True,
                                 max_df=0.95,
                                 min_df=0.05,
                                 max_features=500)
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()
    tracker.stop_task("Vectorize Text")
    return X_train_feats, X_test_feats, feature_names, vectorizer

def train_classifier(X_train_feats, y_train, tracker):
    tracker.start_task("Train Classifier")
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)
    tracker.stop_task("Train Classifier")
    return classifier

def evaluate_classifier(classifier, X_test_feats, y_test, vectorizer, tracker):
    tracker.start_task("Evaluate Classifier")
    y_pred = classifier.predict(X_test_feats)
    clf.show_features(vectorizer, y_test, classifier, n=20)
    metrics.ConfusionMatrixDisplay.from_estimator(classifier, X_test_feats, y_test, cmap=plt.cm.Blues, labels=["FAKE", "REAL"])
    classifier_metrics = metrics.classification_report(y_test, y_pred, output_dict=True)
    tracker.stop_task("Evaluate Classifier")
    return classifier_metrics

def save_model(classifier, vectorizer, output_path, tracker):
    tracker.start_task("Save Model")
    dump(classifier, "models/LR_classifier.joblib")
    dump(vectorizer, "models/LR_tfidf_vectorizer.joblib")
    tracker.stop_task("Save Model")

def plot_learning_curve(X, y, tracker):
    tracker.start_task("Plot Learning Curve")
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = LogisticRegression(random_state=42)
    clf.plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
    tracker.stop_task("Plot Learning Curve")

def train_lrc(input_path, output_path, tracker):
    data = load_data(input_path, tracker)
    X_train, X_test, y_train, y_test = preprocess_data(data, tracker)
    X_train_feats, X_test_feats, _, vectorizer = vectorize_text(X_train, X_test, tracker)
    classifier = train_classifier(X_train_feats, y_train, tracker)
    classifier_metrics = evaluate_classifier(classifier, X_test_feats, y_test, vectorizer, tracker)
    df = pd.DataFrame(classifier_metrics).transpose()
    df.to_csv(os.path.join(output_path, "lrc_report.csv"))
    plot_learning_curve(vectorizer.fit_transform(data["text"]), data["label"], tracker)
    save_model(classifier, vectorizer, output_path, tracker)

def eval_sentence(sentence):
    loaded_classifier = load("models/LR_classifier.joblib")
    loaded_vect = load("models/LR_tfidf_vectorizer.joblib")
    test_sentence = loaded_vect.transform([sentence])
    print(f'The sentence you typed was classified as {loaded_classifier.predict(test_sentence)}')

def main(sentence):
    input_path = os.path.join("in", "fake_or_real_news.csv")
    output_path = "out"

    tracker = EmissionsTracker(project_name="Fake news classification",
                                experiment_id="logistic_regression",
                                output_dir=output_path,
                                output_file="logistic_regression_emissions.csv",
                                log_level="error")

    if os.path.exists(os.path.join("models", "LR_classifier.joblib")):
        eval_sentence(sentence)
    else:
        train_lrc(input_path, output_path, tracker)
        eval_sentence(sentence)

    tracker.stop()
    print(f'Saved emissions report to {os.path.join(output_path, "logistic_regression_emissions.csv")}')

if __name__ == "__main__":
    main()
