import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from load_data import load_data
from preprocess import clean_text
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_test, y_pred, save_path="../outputs/confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def plot_roc_curve(y_test, y_scores, save_path="../outputs/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def evaluate(model_path: str = "../models/classifier.joblib",
             data_path: str = "../data/spam.csv"):
    pipeline = joblib.load(model_path)

    df = load_data(data_path)
    df['text'] = df['text'].apply(clean_text)
    X = df['text']
    y = df['label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_test)

    # ROC needs probability scores — use decision_function if no predict_proba
    clf = pipeline.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        y_scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_scores = pipeline.decision_function(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_scores)


if __name__ == "__main__":
    evaluate()
