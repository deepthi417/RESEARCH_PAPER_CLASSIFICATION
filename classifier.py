"""
Phase 2 classifier wrapper. Loads the already-trained, already-validated
TF-IDF vectorizer + SGDClassifier from disk. Does NOT retrain anything.
"""

import os
import pickle
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'sgd_classifier.pkl')


class ClassifierLoadError(Exception):
    """Raised when the model files are missing or invalid, so the caller
    (app.py) can show a clear, specific error instead of a raw traceback."""
    pass


def load_classifier():
    """
    Returns (vectorizer, clf). Raises ClassifierLoadError with a clear
    message if the model files are missing — never silently falls back
    to an untrained/fake model.
    """
    if not os.path.exists(VECTORIZER_PATH):
        raise ClassifierLoadError(
            f"Classifier vectorizer not found at '{VECTORIZER_PATH}'. "
            f"Run export_models.py in your training notebook first, then "
            f"copy the resulting models/ folder into this project."
        )
    if not os.path.exists(CLASSIFIER_PATH):
        raise ClassifierLoadError(
            f"Classifier model not found at '{CLASSIFIER_PATH}'. "
            f"Run export_models.py in your training notebook first, then "
            f"copy the resulting models/ folder into this project."
        )

    try:
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(CLASSIFIER_PATH, 'rb') as f:
            clf = pickle.load(f)
    except Exception as e:
        raise ClassifierLoadError(f"Failed to load model files — they may be corrupted: {e}")

    if not hasattr(vectorizer, 'vocabulary_'):
        raise ClassifierLoadError("Loaded vectorizer is not fitted (missing vocabulary_).")
    if not hasattr(clf, 'classes_'):
        raise ClassifierLoadError("Loaded classifier is not fitted (missing classes_).")

    return vectorizer, clf


def classify(title, abstract, vectorizer, clf, top_n=2):
    """
    Returns the same structure Phase 4 used: primary prediction + runner-up.
    """
    text = f"{title.strip()} {abstract.strip()}"
    text_vec = vectorizer.transform([text])

    probs = clf.predict_proba(text_vec)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    top_classes = clf.classes_[top_idx]
    top_probs = probs[top_idx]

    result = {
        'primary_category': str(top_classes[0]),
        'confidence': round(float(top_probs[0]), 4),
        'runner_up': {
            'category': str(top_classes[1]),
            'confidence': round(float(top_probs[1]), 4)
        } if len(top_classes) > 1 else None
    }
    return result
