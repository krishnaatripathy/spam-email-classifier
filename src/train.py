import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from load_data import load_data
from preprocess import clean_text, build_vectorizer


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ('tfidf', build_vectorizer()),
        ('clf', model)
    ])


def train(data_path: str = "../data/spam.csv"):
    df = load_data(data_path)
    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Naive Bayes":        MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM":                 LinearSVC(max_iter=1000)
    }

    results = {}
    best_name, best_pipeline, best_score = None, None, 0

    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        results[name] = score
        print(f"{name:25s} accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_name} ({best_score:.4f})")

    os.makedirs("../models", exist_ok=True)
    joblib.dump(best_pipeline, "../models/classifier.joblib")
    print("Model saved to models/classifier.joblib")

    return best_pipeline, X_test, y_test, results


if __name__ == "__main__":
    train()
