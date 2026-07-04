"""
feature_extraction.py
----------------------
Converts cleaned text into numerical feature vectors that a machine-learning
model can consume. Two approaches are implemented and compared:

    1. CountVectorizer  -> Bag-of-Words: raw word frequency counts.
    2. TfidfVectorizer  -> Term Frequency - Inverse Document Frequency:
                           weighs words by how distinctive they are, not
                           just how often they appear.

TF-IDF generally performs better for spam detection because it down-weights
very common words (which appear in both spam and ham) and up-weights rarer,
more discriminative words (e.g. "lottery", "urllink", "free").
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_count_vectorizer(max_features: int = 3000) -> CountVectorizer:
    """Bag-of-Words vectorizer: counts raw word occurrences."""
    return CountVectorizer(max_features=max_features, ngram_range=(1, 2))


def get_tfidf_vectorizer(max_features: int = 3000) -> TfidfVectorizer:
    """TF-IDF vectorizer: weights words by importance/rarity across documents."""
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))


def fit_transform_features(train_texts, test_texts, method: str = "tfidf",
                            max_features: int = 3000):
    """
    Fits a vectorizer on the training texts and transforms both train and
    test sets. Returns the transformed matrices and the fitted vectorizer
    (needed later for the Streamlit app / prediction module).

    Parameters
    ----------
    train_texts, test_texts : iterable of str
        Cleaned email text.
    method : str
        "tfidf" (default) or "count".
    max_features : int
        Vocabulary size cap.

    Returns
    -------
    X_train, X_test, vectorizer
    """
    if method == "count":
        vectorizer = get_count_vectorizer(max_features)
    else:
        vectorizer = get_tfidf_vectorizer(max_features)

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


if __name__ == "__main__":
    sample_train = ["free money now click here", "meeting scheduled tomorrow morning"]
    sample_test = ["click here for free prize"]

    X_train, X_test, vec = fit_transform_features(sample_train, sample_test, method="tfidf")
    print("Vocabulary size:", len(vec.vocabulary_))
    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
