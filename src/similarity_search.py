"""
Phase 7 — Similarity Search. Independent of the 63-class classifier —
takes only raw title+abstract text, never a predicted class. No LLM.

This includes the self-match fix validated during Phase 7 (title-match
alone is sufficient to exclude the query paper from its own results —
requiring an additional similarity threshold was found to be fragile).
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from src.taxonomy import get_domain_info

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'arxiv_embeddings.npy')
META_PATH = os.path.join(DATA_DIR, 'arxiv_embeddings_meta.jsonl')

DEFAULT_K = 5
RETRIEVAL_BUFFER_MULTIPLIER = 3


class SimilaritySearchService:

    def __init__(self):
        self.model = None
        self.index = None
        self.meta = None
        self.status = 'not_loaded'
        self.error_message = None

    def load(self):
        if not os.path.exists(EMBEDDINGS_PATH):
            self.status = 'unavailable'
            self.error_message = (
                f"Embeddings file not found at '{EMBEDDINGS_PATH}'. "
                f"Copy your Phase 3 arxiv_embeddings.npy into the data/ folder."
            )
            return
        if not os.path.exists(META_PATH):
            self.status = 'unavailable'
            self.error_message = (
                f"Metadata file not found at '{META_PATH}'. "
                f"Copy your Phase 3 arxiv_embeddings_meta.jsonl into the data/ folder."
            )
            return

        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            embeddings = np.load(EMBEDDINGS_PATH).astype('float32')
            faiss.normalize_L2(embeddings)

            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)

            self.meta = []
            with open(META_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    self.meta.append(json.loads(line))

            assert self.index.ntotal == len(self.meta), (
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs {len(self.meta)} metadata rows"
            )
            self.status = 'ok'
        except Exception as e:
            self.status = 'unavailable'
            self.error_message = f"Similarity search failed to load: {e}"

    def _embed_query(self, title, abstract):
        text = f"{title.strip()} {abstract.strip()}"
        vec = self.model.encode([text]).astype('float32')
        faiss.normalize_L2(vec)
        return vec

    def search(self, title, abstract, k=DEFAULT_K):
        if self.status != 'ok':
            return {
                'similar_papers': [],
                'query_excluded_self_match': False,
                'status': 'unavailable',
                'error_message': self.error_message
            }

        query_vec = self._embed_query(title, abstract)
        search_k = min(k * RETRIEVAL_BUFFER_MULTIPLIER, self.index.ntotal)

        raw_scores, indices = self.index.search(query_vec, search_k)
        raw_scores, indices = raw_scores[0], indices[0]

        query_title_norm = ' '.join(title.strip().lower().split())

        results = []
        seen_titles = set()
        self_excluded = False

        for raw_score, idx in zip(raw_scores, indices):
            if idx < 0:
                continue

            record = self.meta[idx]
            result_title_norm = ' '.join(record['title'].strip().lower().split())

            # Self-match: exact normalized title match alone is sufficient.
            # (An additional similarity-threshold requirement was tested and
            # found fragile — a query missing its abstract, or with minor
            # whitespace/encoding differences, can legitimately score below
            # a strict threshold even for the exact same paper.)
            if result_title_norm == query_title_norm:
                self_excluded = True
                continue

            if result_title_norm in seen_titles:
                continue
            seen_titles.add(result_title_norm)

            broad_domain, subdomain = get_domain_info(record['primary_category'])

            raw_cosine_similarity = round(float(raw_score), 4)
            ui_similarity_score = round(float(raw_score) * 100, 1)

            results.append({
                'title': record['title'],
                'broad_domain': broad_domain,
                'subdomain': subdomain,
                'primary_category': record['primary_category'],
                'raw_cosine_similarity': raw_cosine_similarity,
                'similarity_score': ui_similarity_score,
                'explanation': (
                    "Shares strong semantic overlap in topic and terminology with the query, "
                    "based on embedding similarity."
                    if ui_similarity_score >= 70 else
                    "Shares some topical overlap with the query, based on embedding similarity."
                )
            })

            if len(results) >= k:
                break

        return {
            'similar_papers': results,
            'query_excluded_self_match': self_excluded,
            'status': 'ok',
            'error_message': None
        }
