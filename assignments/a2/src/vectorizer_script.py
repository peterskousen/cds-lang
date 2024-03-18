import os
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize():
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
    #return X, y, X_train, X_test, y_train, y_test, vectorizer, X_train_feats, X_test_feats, feature_names