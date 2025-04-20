##!/usr/bin/env python3

import os
import logging
import time
import random
import json
import pickle
import gc
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm



from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher

# ANN & Embedding
import faiss
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import Pipeline

# Optional dataset libraries
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import ir_datasets
except ImportError:
    ir_datasets = None

# ==============================================================================
# Logging setup
# ==============================================================================
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================
K_VALUES = [1, 5, 10, 20, 50, 100]

ID_QUERY_LIMIT = 1000
RARE_QUERY_LIMIT = 1000
QUERY_LIMIT = 1000

DATASET_CONFIG = {
    "homedepot": {
        "enabled": True,
        "loader": "load_home_depot_data",
        "params": {
            "train_csv": "/home/alessandro/data/home-depot-product-search-relevance/train.csv.zip",
            "rare_term_max_docs": 5,
            "id_query_limit": ID_QUERY_LIMIT,
            "rare_query_limit": RARE_QUERY_LIMIT,
            "query_limit": QUERY_LIMIT,
        },
        "run_query_sets": ['id', 'standard', 'rare']
    },
    "msmarco_passage": {
        "enabled": True,
        "loader": "load_msmarco_passage_data",
        "params": {"split": "validation",
                   "id_query_limit": ID_QUERY_LIMIT,
                   "rare_query_limit": RARE_QUERY_LIMIT,
                   "query_limit": QUERY_LIMIT},
        "run_query_sets": ['id', 'standard', 'rare'],

    },
    "trec_covid": {
        "enabled": True,
        "loader": "load_trec_covid_data",
        "params": {
            "id_query_limit": ID_QUERY_LIMIT,
            "rare_query_limit": RARE_QUERY_LIMIT,
            "query_limit": QUERY_LIMIT,
        },
        "run_query_sets": ['id', 'standard', 'rare'],
    }
}

SWEEP_CONFIG = {
    "embedding_dims": [256, 512, 1024], #, 512, 768],
    "trigram_modes": ['across_tokens', 'within_token'],
    "ivf_nlists": [200,  500, 1000],
    "ivf_nprobe_percentages": [0.1, 0.2, 0.5],
    "hnsw_m_values": [ 64],
    "hnsw_ef_construction": [100, 200],
    "hnsw_ef_search": [ 100],
    "base_random_state": 42,
    "base_density": 0.9,
    "base_n_features_sparse": 2 ** 18,
}

SEARCH_BATCH_SIZE = 256

# ==============================================================================
# Utility functions
# ==============================================================================

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle to {path}")


def save_json(data: Any, filepath: Union[str, Path]) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {path}")


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ==============================================================================
# Data loaders with logging
# ==============================================================================

def load_home_depot_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load Home Depot product search relevance data, and build three query sets:
      - 'id': a random sample of product_uid queries (with golden doc = itself)
      - 'standard': the provided customer search_term → product_uid relevance judgments,
                    capped by params['query_limit']
      - 'rare': generated rare‐term queries (unigrams/bigrams appearing in <= rare_term_max_docs)

    Expects params to contain:
      - train_csv: path to train.csv.zip or train.csv
      - id_query_limit: maximum number of ID queries to sample
      - query_limit: maximum number of standard search_term queries to use
      - rare_term_max_docs: int threshold for rare terms
      - rare_query_limit: maximum number of rare queries to sample
    """
    logger.info("Loading Home Depot data from %s", params['train_csv'])
    df = pd.read_csv(
        params['train_csv'],
        usecols=['product_uid', 'product_title', 'search_term'],
        encoding='ISO-8859-1'
    )
    df['product_uid'] = df['product_uid'].astype(str)
    df['search_term'] = df['search_term'].fillna('').astype(str)

    # Build corpus of unique products
    df_prod = df[['product_uid', 'product_title']] \
        .drop_duplicates('product_uid') \
        .set_index('product_uid')
    corpus = [
        {'doc_id': uid, 'text': row['product_title'] or ''}
        for uid, row in df_prod.iterrows()
    ]
    logger.info("Loaded %d unique products into corpus", len(corpus))

    query_sets: Dict[str, Tuple[Dict[str, str], Dict[str, Set[str]]]] = {}

    # 1) ID queries (random sample)
    all_ids = [d['doc_id'] for d in corpus]
    id_limit = params.get('id_query_limit', len(all_ids))
    if id_limit < len(all_ids):
        sampled_ids = random.sample(all_ids, id_limit)
        logger.info("Sampling %d ID queries out of %d", id_limit, len(all_ids))
    else:
        sampled_ids = all_ids
        logger.info("Using all %d ID queries", len(all_ids))
    queries_id = {i: i for i in sampled_ids}
    qrels_id   = {i: {i} for i in sampled_ids}
    query_sets['id'] = (queries_id, qrels_id)

    # 2) Standard customer search_term queries (with cap)
    grouped = df.groupby('search_term')['product_uid'] \
                .agg(lambda u: set(u.astype(str)))
    all_terms = [term for term in grouped.index if term]
    qlim = params.get('query_limit', len(all_terms))
    if qlim < len(all_terms):
        chosen_terms = random.sample(all_terms, qlim)
        logger.info("Sampling %d standard queries out of %d", qlim, len(all_terms))
    else:
        chosen_terms = all_terms
        logger.info("Using all %d standard queries", len(all_terms))
    queries_std = {term: term for term in chosen_terms}
    qrels_std   = {term: grouped[term] for term in chosen_terms}
    query_sets['standard'] = (queries_std, qrels_std)

    # 3) Rare‐term queries
    rare_q, rare_qr = generate_rare_term_queries(
        corpus,
        params['rare_term_max_docs'],
        params['rare_query_limit']
    )
    logger.info("Prepared 'rare' query set (%d queries)", len(rare_q))
    query_sets['rare'] = (rare_q, rare_qr)

    return {
        'corpus': corpus,
        'query_sets': query_sets
    }



def load_msmarco_passage_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loader for MS MARCO Passage v1.1 (HF schema):
      - 'id': random sample of passage‐IDs as queries (self‐relevance)
      - 'standard': true query→passage relevance from passages.is_selected
      - 'rare': generated rare‐term queries

    Expects in params:
      - split            : HF split name (e.g. 'validation')
      - id_query_limit   : max # of ID queries to sample
      - rare_term_max_docs: threshold for rare‐term generation
    """
    if load_dataset is None:
        raise ImportError("Install 'datasets' to load MS MARCO.")
    logger.info("Loading MS MARCO v1.1 split=%s", params.get('split'))
    ds = load_dataset('ms_marco', 'v1.1', split=params.get('split', 'validation'))

    corpus: List[Dict[str,str]] = []
    std_q:  Dict[str,str]          = {}
    std_qr: Dict[str,Set[str]]     = defaultdict(set)

    logger.info("Flattening query‐passage pairs...")
    for d in tqdm(ds, desc="MSMARCO load"):
        qid   = str(d['query_id'])
        qtext = d['query']
        std_q[qid] = qtext

        passages = d['passages']
        texts    = passages['passage_text']
        selected = passages['is_selected']

        # flatten each passage
        for idx, (txt, sel) in enumerate(zip(texts, selected)):
            pid = f"{qid}_p{idx}"
            corpus.append({'doc_id': pid, 'text': txt})
            if sel == 1:
                std_qr[qid].add(pid)

    logger.info("Built corpus of %d passages for %d queries", len(corpus), len(std_q))

    # prepare query_sets
    query_sets: Dict[str,Tuple[Dict[str,str],Dict[str,Set[str]]]] = {}

    # 1) ID queries (sampled)
    all_ids = [d['doc_id'] for d in corpus]
    limit   = params.get('id_query_limit', len(all_ids))
    if limit < len(all_ids):
        sampled = random.sample(all_ids, limit)
        logger.info("Sampling %d ID‐queries from %d passages", limit, len(all_ids))
    else:
        sampled = all_ids
        logger.info("Using all %d ID‐queries", len(all_ids))
    q_id = {pid: pid for pid in sampled}
    qr_id= {pid: {pid} for pid in sampled}
    query_sets['id'] = (q_id, qr_id)

    # 2) Standard queries
    # Optionally cap number of queries
    all_qids = list(std_q)
    cap = params.get('query_limit', len(all_qids))
    chosen = all_qids[:cap]
    q_std  = {qid: std_q[qid] for qid in chosen}
    qr_std = {qid: std_qr[qid] for qid in chosen}
    logger.info("Using %d standard queries", len(q_std))
    query_sets['standard'] = (q_std, qr_std)

    # 3) Rare‐term queries
    rare_q, rare_qr = generate_rare_term_queries(
        corpus,
        params.get('rare_term_max_docs', 5), params.get('rare_query_limit')
    )
    logger.info("Generated %d rare-term queries", len(rare_q))
    query_sets['rare'] = (rare_q, rare_qr)

    return {'corpus': corpus, 'query_sets': query_sets}


def load_trec_covid_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loader for TREC‑COVID via ir_datasets:
      - 'id': random sample of doc IDs as queries (requires id_query_limit)
      - 'standard': official topic→doc relevance judgments (capped by query_limit)
      - 'rare': generated rare‐term queries (threshold rare_term_max_docs)

    Expects params:
      - id_query_limit: int (max # of ID queries to sample)
      - query_limit: int (max # of standard topics to use)
      - rare_term_max_docs: int (threshold for rare terms)
    """
    if ir_datasets is None:
        raise ImportError("Install 'ir_datasets' to load TREC‑COVID.")
    logger.info("Loading TREC‑COVID documents via ir_datasets.load('cord19/trec-covid')…")
    ds = ir_datasets.load("cord19/trec-covid")

    # Build corpus from abstracts (Cord19Doc has fields: doc_id, title, doi, date, abstract)
    corpus = [
        {'doc_id': str(d.doc_id), 'text': d.abstract or ''}
        for d in ds.docs_iter()
    ]
    logger.info("Loaded %d documents (abstracts only)", len(corpus))

    # Standard topics & qrels
    std_q  = {str(q.query_id): q.title for q in ds.queries_iter()}
    std_qr = defaultdict(set)
    for r in ds.qrels_iter():
        std_qr[str(r.query_id)].add(str(r.doc_id))
    logger.info("Loaded %d standard topics", len(std_q))

    # Prepare query_sets
    qs: Dict[str, Tuple[Dict[str, str], Dict[str, Set[str]]]] = {}

    # 1) ID queries (sample)
    all_ids = [d['doc_id'] for d in corpus]
    id_limit = params.get('id_query_limit', len(all_ids))
    if id_limit < len(all_ids):
        sampled = random.sample(all_ids, id_limit)
        logger.info("Sampling %d ID queries from %d docs", id_limit, len(all_ids))
    else:
        sampled = all_ids
        logger.info("Using all %d ID queries", len(all_ids))
    q_id = {i: i for i in sampled}
    qr_id = {i: {i} for i in sampled}
    qs['id'] = (q_id, qr_id)

    # 2) Standard topic queries (possibly limited)
    topic_ids = list(std_q)
    qlim = params.get('query_limit', len(topic_ids))
    chosen = topic_ids[:qlim]
    q_std  = {qid: std_q[qid]  for qid in chosen}
    qr_std = {qid: std_qr[qid] for qid in chosen}
    logger.info("Using %d standard topics", len(q_std))
    qs['standard'] = (q_std, qr_std)

    # 3) Rare‐term queries
    rare_q, rare_qr = generate_rare_term_queries(
        corpus,
        params.get('rare_term_max_docs', 5), params['rare_query_limit']
    )
    logger.info("Generated %d rare‐term queries", len(rare_q))
    qs['rare'] = (rare_q, rare_qr)

    return {'corpus': corpus, 'query_sets': qs}




def generate_rare_term_queries(
        docs: List[Dict[str, Any]],
        rare_max_docs: int,
        query_limit: Optional[int] = 1000
) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    term_to_docs: Dict[str, Set[str]] = defaultdict(set)
    pre = [preprocess_text(d['text']) for d in docs]
    ids = [d['doc_id'] for d in docs]
    for idx, txt in enumerate(pre):
        words = txt.split()
        for w in words:
            if len(w) > 3:
                term_to_docs[f'unigram:{w}'].add(ids[idx])
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i + 1]}"
            term_to_docs[f'bigram:{bg}'].add(ids[idx])
    rare = {t: ds for t, ds in term_to_docs.items()
            if 1 <= len(ds) <= rare_max_docs}
    queries, qrels = {}, {}
    for term, ds in rare.items():
        ttype, ttxt = term.split(':', 1)
        qid = f"{ttype}_{ttxt}"
        queries[qid] = ttxt
        qrels[qid] = ds
    if query_limit and query_limit < len(queries):
        keys = random.sample(list(queries), query_limit)
        queries = {k: queries[k] for k in keys}
        qrels = {k: qrels[k] for k in keys}
    return queries, qrels


# ==============================================================================
# Embedder & ANN Index
# ==============================================================================
class TrigramRPEmbedder:
    def __init__(
            self, n_components_dense: int, trigram_mode: str,
            n_features_sparse: int, density: float,
            random_state: int
    ):
        analyzer = 'char_wb' if trigram_mode == 'within_token' else 'char'
        self.pipeline = Pipeline([
            ('hasher', HashingVectorizer(
                n_features=n_features_sparse,
                analyzer=analyzer,
                ngram_range=(3, 3),
                norm=None,
                alternate_sign=False
            )),
            ('proj', SparseRandomProjection(
                n_components=n_components_dense,
                density=density,
                random_state=random_state,
                dense_output=True
            ))
        ])
        self.embedding_dim = n_components_dense
        self.is_fitted = False

    def encode_corpus(
            self, texts: List[str],
            cache_dir: Optional[Path] = None,
            cache_prefix: str = 'trigram_rp'
    ) -> np.ndarray:
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            emb_path = cache_dir / f"{cache_prefix}_emb_{self.embedding_dim}.npy"
            pipe_path = cache_dir / f"{cache_prefix}_pipe_{self.embedding_dim}.pkl"
            if emb_path.exists() and pipe_path.exists():
                embs = np.load(emb_path)
                with open(pipe_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                self.is_fitted = True
                return embs
        X = self.pipeline.fit_transform(texts).astype(np.float32)
        self.is_fitted = True
        if cache_dir:
            np.save(emb_path, X)
            with open(pipe_path, 'wb') as f:
                pickle.dump(self.pipeline, f)
        return X

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Embedder not fitted")
        return self.pipeline.transform(texts).astype(np.float32)


def normalize_l2(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / (norms + 1e-8)




def evaluate_single_run(
    run_name: str,
    query_results: Dict[str, List[Tuple[str, float]]],
    queries: Dict[str, str],
    qrels: Dict[str, Set[str]],
    k_values: List[int]
) -> Dict[str, Any]:
    """
    Aggregates per-query recall metrics into a single summary for one run.

    Args:
        run_name: Identifier for this experiment configuration.
        query_results: Mapping from query ID to list of (doc_id, score) tuples.
        queries: Mapping from query ID to the query text (unused here but kept for symmetry).
        qrels: Mapping from query ID to the set of relevant document IDs.
        k_values: List of k cutoffs for Recall@k.

    Returns:
        A dict with keys:
          - "run_name": the run_name argument
          - "metrics": a dict mapping "Recall@k" to the mean recall across queries
    """
    per_query_metrics: List[Dict[str, float]] = []

    for qid, retrieved in query_results.items():
        relevant = qrels.get(qid, set())
        # Skip queries with no retrieved docs or no relevance labels
        if not retrieved or not relevant:
            continue
        # Compute Recall@k for this query
        q_metrics = evaluate_query_results(qid, retrieved, relevant, k_values)
        per_query_metrics.append(q_metrics)

    # If no queries could be evaluated, return empty metrics
    if not per_query_metrics:
        return {"run_name": run_name, "metrics": {}}

    # Aggregate by averaging across queries
    aggregated = aggregate_metrics(per_query_metrics)
    return {"run_name": run_name, "metrics": aggregated}

class ANNIndex:
    def __init__(
            self, embedding_dim: int,
            metric: str = 'l2',
            index_type: str = 'flat',
            ivf_nlist: Optional[int] = None,
            ivf_nprobe: Optional[int] = None,
            hnsw_m: Optional[int] = None,
            hnsw_ef_construction: Optional[int] = None,
            hnsw_ef_search: Optional[int] = None
    ):
        metric_type = faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT
        if index_type == 'flat':
            self.index = faiss.IndexFlat(embedding_dim, metric_type)
            self.is_trained = True
        elif index_type == 'ivf':
            quant = faiss.IndexFlat(embedding_dim, metric_type)
            self.index = faiss.IndexIVFFlat(quant, embedding_dim, ivf_nlist, metric_type)
            self.index.nprobe = ivf_nprobe
            self.is_trained = False
        elif index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m, metric_type)
            self.index.hnsw.efConstruction = hnsw_ef_construction
            self.index.hnsw.efSearch = hnsw_ef_search
            self.is_trained = True
        else:
            raise ValueError(f"Unknown index type {index_type}")
        self.is_built = False

    def train(self, emb: np.ndarray) -> None:
        if hasattr(self.index, 'train') and not self.is_trained:
            self.index.train(emb)
            self.is_trained = True

    def add(self, emb: np.ndarray) -> None:
        if not self.is_trained:
            return
        self.index.add(emb)
        self.is_built = True

    def search(
            self, q_emb: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_built:
            return np.empty((q_emb.shape[0], k)), np.full((q_emb.shape[0], k), -1, dtype=int)
        k2 = min(k, self.index.ntotal)
        D, I = self.index.search(q_emb, k2)
        return D, I


# ==============================================================================
# Evaluation & Search
# ==============================================================================
def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 1.0
    ret_k = retrieved[:k]
    return len(set(ret_k) & relevant) / len(relevant)


def evaluate_query_results(
        qid: str,
        retrieved: List[Tuple[str, float]],
        relevant: Set[str],
        k_values: List[int]
) -> Dict[str, float]:
    ids = [doc for doc, _ in retrieved]
    return {f"Recall@{k}": recall_at_k(relevant, ids, k) for k in k_values}


def aggregate_metrics(all_q: List[Dict[str, float]]) -> Dict[str, float]:
    agg: Dict[str, List[float]] = defaultdict(list)
    for m in all_q:
        for k, v in m.items():
            agg[k].append(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def run_search_queries_ann(
    queries: Dict[str, str],
    index: ANNIndex,
    k: int,
    doc_ids: List[str],
    embedder: TrigramRPEmbedder,
    batch_size: int = SEARCH_BATCH_SIZE
) -> Dict[str, List[Tuple[str, float]]]:
    logger.info(f"Starting ANN search (index={index}, k={k}) for {len(queries)} queries...")
    results: Dict[str, List[Tuple[str, float]]] = {}
    qids = list(queries)
    texts = [preprocess_text(queries[q]) for q in qids]
    for i in tqdm(range(0, len(texts), batch_size), desc='ANN search'):
        batch_ids = qids[i:i+batch_size]
        batch_txt = texts[i:i+batch_size]
        emb = embedder.encode_batch(batch_txt)
        emb = normalize_l2(emb)
        D, I = index.search(emb, k)
        for j, qid in enumerate(batch_ids):
            docs: List[Tuple[str, float]] = []
            for dist, idx in zip(D[j], I[j]):
                if idx < 0:
                    continue
                docid = doc_ids[idx]
                score = 1.0 / (1.0 + dist)
                docs.append((docid, score))
            results[qid] = docs
    logger.info("ANN search completed.")
    return results


# ==============================================================================
# Sweep runner
# ==============================================================================
def load_existing_results(filepath):
    if filepath.exists():
        with open(filepath, 'r') as f:
            existing_results = json.load(f)
        existing_runs = {res['run_name'] for res in existing_results}
        return existing_results, existing_runs
    else:
        return [], set()

# Save incremental results

def save_incremental_result(filepath, result):
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# Update run_sweep to check for existing runs and save incrementally

def run_sweep(name, data, output_dir, run_sets):
    corpus = data['corpus']
    doc_ids = [d['doc_id'] for d in corpus]
    raw_texts = [preprocess_text(d['text']) for d in corpus]

    results_file = output_dir / "sweep_results.json"
    results, existing_runs = load_existing_results(results_file)

    max_k = max(K_VALUES)

    for qset_name in run_sets:
        queries, qrels = data['query_sets'][qset_name]

        if qset_name == 'id':
            corpus_texts = [
                preprocess_text(f"docid {d['doc_id']} {d['text']}")
                for d in corpus
            ]
        else:
            corpus_texts = raw_texts

        proc_queries = {qid: preprocess_text(q) for qid, q in queries.items()}

        for mode in SWEEP_CONFIG['trigram_modes']:
            for dim in SWEEP_CONFIG['embedding_dims']:
                embedder = TrigramRPEmbedder(
                    n_components_dense=dim,
                    trigram_mode=mode,
                    n_features_sparse=SWEEP_CONFIG['base_n_features_sparse'],
                    density=SWEEP_CONFIG['base_density'],
                    random_state=SWEEP_CONFIG['base_random_state']
                )
                emb = embedder.encode_corpus(corpus_texts)
                emb_norm = normalize_l2(emb)

                # Flat
                run_name = f"{name}.{qset_name}.Flat_d{dim}_{mode}"
                if run_name not in existing_runs:
                    idx_flat = ANNIndex(dim, 'l2', 'flat')
                    idx_flat.add(emb_norm)
                    res_flat = run_search_queries_ann(proc_queries, idx_flat, max_k, doc_ids, embedder)
                    result = evaluate_single_run(run_name, res_flat, queries, qrels, K_VALUES)
                    save_incremental_result(results_file, result)
                else:
                    print('skipping ', run_name)
                # IVF
                for nlist in SWEEP_CONFIG['ivf_nlists']:
                    if emb_norm.shape[0] < nlist:
                        continue
                    for perc in SWEEP_CONFIG['ivf_nprobe_percentages']:
                        npv = min(nlist, max(1, math.ceil(nlist * perc)))
                        rn = f"{name}.{qset_name}.IVF_d{dim}_{mode}_nl{nlist}_np{npv}"
                        if rn in existing_runs:
                            print('skipping: ', rn)
                            continue
                        idx_ivf = ANNIndex(dim, 'l2', 'ivf', ivf_nlist=nlist, ivf_nprobe=npv)
                        idx_ivf.train(emb_norm)
                        idx_ivf.add(emb_norm)
                        if idx_ivf.is_built:
                            r = run_search_queries_ann(proc_queries, idx_ivf, max_k, doc_ids, embedder)
                            result = evaluate_single_run(rn, r, queries, qrels, K_VALUES)
                            save_incremental_result(results_file, result)

                # HNSW
                for m in SWEEP_CONFIG['hnsw_m_values']:
                    for efC in SWEEP_CONFIG['hnsw_ef_construction']:
                        for efS in SWEEP_CONFIG['hnsw_ef_search']:
                            rn = f"{name}.{qset_name}.HNSW_d{dim}_{mode}_M{m}_efC{efC}_efS{efS}"
                            if rn in existing_runs:
                                print('skipping: ', rn)
                                continue
                            idx_h = ANNIndex(dim, 'l2', 'hnsw', hnsw_m=m, hnsw_ef_construction=efC, hnsw_ef_search=efS)
                            idx_h.add(emb_norm)
                            if idx_h.is_built:
                                r = run_search_queries_ann(proc_queries, idx_h, max_k, doc_ids, embedder)
                                result = evaluate_single_run(rn, r, queries, qrels, K_VALUES)
                                save_incremental_result(results_file, result)

        # BM25
        bm25_run_name = f"{name}.{qset_name}.BM25"
        if bm25_run_name not in existing_runs:
            path_out_dir = Path(str(output_dir / f"bm25_{qset_name}"))
            path_out_dir.mkdir(parents=True, exist_ok=True)
            path_out_dir = Path(str(output_dir / f"bm25_{qset_name}" / "index"))
            path_out_dir.mkdir(parents=True, exist_ok=True)

            bm25_searcher = LuceneSearcher(str(output_dir / f"bm25_{qset_name}" / "index"))
            bm25_searcher.set_bm25()

            bm25_results = {}
            for qid, qtext in proc_queries.items():
                hits = bm25_searcher.search(qtext, max_k)
                bm25_results[qid] = [(hit.docid, hit.score) for hit in hits]

            result = evaluate_single_run(bm25_run_name, bm25_results, queries, qrels, K_VALUES)
            save_incremental_result(results_file, result)
        else:
            print('skipping ', bm25_run_name)

    return results


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    base_out = Path('./output/sweep_all')
    all_res: List[Dict[str, Any]] = []

    for ds_name, ds_cfg in DATASET_CONFIG.items():
        if not ds_cfg['enabled']:
            continue
        logger.info(f"=== Processing dataset: {ds_name} ===")
        loader = globals()[ds_cfg['loader']]
        data = loader(ds_cfg['params'])

        # **Pass your query sets here**
        run_sets = ds_cfg.get('run_query_sets', [])
        out_dir = base_out / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)

        res = run_sweep(ds_name, data, out_dir, run_sets)
        all_res.extend(res)

    save_json(all_res, base_out / 'all_sweep_results.json')
    logger.info("All sweeps completed.")
