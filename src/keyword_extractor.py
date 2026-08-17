"""
Phase 4 — KeyBERT-based keyword extraction. Reuses the same
sentence-transformer already used for classification-comparison (Phase 3)
and similarity search (Phase 7), so no extra model download.
"""

from keybert import KeyBERT

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_keyword_model():
    return KeyBERT(model=EMBEDDING_MODEL_NAME)


def extract_keywords(kw_model, title, abstract, top_n=8):
    text = f"{title.strip()} {abstract.strip()}"
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n,
        use_mmr=True,
        diversity=0.5
    )
    return [kw for kw, score in keywords]
