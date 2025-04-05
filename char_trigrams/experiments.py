import os
import time
import re
import numpy as np
import pandas as pd
import faiss
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
import random
import collections
import gc  # Garbage collector
import pickle # For caching rarity analysis
import math # For ceiling division in batching

# Try importing ir_measures, set flag
try:
    import ir_measures
    from ir_measures import * # Imports metrics like nDCG, Recall, P, RR
    IR_MEASURES_AVAILABLE = True
except ImportError:
    print("WARNING: ir_measures library not found. Benchmark evaluation will be skipped.")
    print("Install using: pip install ir_measures")
    IR_MEASURES_AVAILABLE = False

# --- Constants and Configuration ---
BASE_OUTPUT_DIR = 'ann_limitation_final_experiments_batched' # Changed output dir name
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Performance Config
SEARCH_BATCH_SIZE = 128 # Process queries in batches for Faiss search

# Experiment 1 Config
MSMARCO_CORPUS_LIMIT = 500000 # Limit corpus size for faster testing (set to None for full)
SBERT_MODEL_NAME = 'msmarco-distilbert-base-v4' # Standard SBERT for MSMARCO
EXPECTED_SBERT_MRR = 0.32 # Approx MRR@10 for msmarco-distilbert-base-v4 on dev small for verification
K_NEIGHBORS_MSMARCO = 100 # Retrieve more for Recall@k eval
# Rarity Config (adjust thresholds as needed)
RARE_TOKEN_MAX_DF = 0.001 # Max document frequency for a token to be rare
RARE_TRIGRAM_MAX_FREQ = 100 # Max corpus-wide count for a trigram to be rare
NUM_RARE_QUERY_SAMPLES = 500 # Limit number of rare queries to evaluate for speed

# Experiment 2 Config
HOME_DEPOT_TRAIN_CSV = '/home/alessandro/data/home-depot-product-search-relevance/train.csv' # !!! UPDATE PATH !!!
HOME_DEPOT_DESC_CSV = '/home/alessandro/data/home-depot-product-search-relevance/product_descriptions.csv' # !!! UPDATE PATH !!! (Optional)
HD_CORPUS_SIZE_LIMIT = 20000 # Limit corpus size for faster testing (set to None for full)
K_NEIGHBORS_HD = 10

# Shared Config
# Trigram+RP Pipeline Config
N_FEATURES_SPARSE = 2**18
N_COMPONENTS_DENSE = 300
# Faiss ANN Params (can be tuned per experiment/embedding if needed)
NLIST_IVF = 100
NPROBE_IVF = 10
HNSW_M = 32
EF_CONSTRUCTION_HNSW = 64
EF_SEARCH_HNSW = 32

# --- Helper Functions ---
# (preprocess_text_simple, preprocess_text_hd, prepare_text_for_encoding_hd)
# (analyze_rarity, find_rare_feature_queries)
# (evaluate_runs)
# ... Keep these helper functions exactly as they were in the previous version ...
def preprocess_text_simple(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_hd(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s\-_]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_text_for_encoding_hd(row, trigram_mode, text_processor):
    base_text = text_processor(row.get('full_text', ''))
    doc_id = str(row.get('doc_id', ''))
    id_text = "id_" + text_processor(doc_id)
    if trigram_mode == 'within_token':
        separator = " _TOKSEP_ "
        tokens = base_text.split()
        processed_text = separator.join(tokens)
        final_text = processed_text + separator + id_text
        return final_text
    elif trigram_mode == 'across_tokens':
        return base_text + " " + id_text
    else:
        raise ValueError(f"Unknown trigram_mode: {trigram_mode}")

def analyze_rarity(corpus_texts, cache_file, token_max_df=0.001, trigram_max_freq=100):
    if os.path.exists(cache_file):
        print(f"Loading cached rarity analysis from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    print("Analyzing corpus for rare features (this may take time)...")
    rare_tokens = set()
    rare_trigrams = set()
    print(" Finding rare tokens...")
    try:
        token_vectorizer = CountVectorizer(stop_words='english', max_df=token_max_df)
        token_vectorizer.fit(corpus_texts)
        rare_tokens = set(token_vectorizer.vocabulary_.keys())
        print(f" Found {len(rare_tokens)} rare tokens (max_df <= {token_max_df}).")
    except Exception as e:
         print(f" Warning: Token rarity analysis failed: {e}. Skipping rare tokens.")
    print(" Finding rare trigrams (using hashing approximation)...")
    try:
        trigram_hasher = HashingVectorizer(n_features=2**20, analyzer='char', ngram_range=(3, 3), norm=None, alternate_sign=False)
        trigram_counts_sparse = trigram_hasher.fit_transform(corpus_texts)
        total_counts_per_hash = np.array(trigram_counts_sparse.sum(axis=0)).flatten()
        rare_hash_indices = np.where(total_counts_per_hash <= trigram_max_freq)[0]
        print(f" Found {len(rare_hash_indices)} feature hashes corresponding to potentially rare trigrams (max_freq <= {trigram_max_freq}).")
        rare_trigram_hash_indices = set(rare_hash_indices)
    except Exception as e:
         print(f" Warning: Trigram rarity analysis failed: {e}. Skipping rare trigrams.")
         rare_trigram_hash_indices = set()
    results = {'rare_tokens': rare_tokens, 'rare_trigram_hash_indices': rare_trigram_hash_indices, 'trigram_hasher': trigram_hasher}
    print(f"Saving rarity analysis to {cache_file}...")
    with open(cache_file, 'wb') as f: pickle.dump(results, f)
    return results

def find_rare_feature_queries(queries_dict, rarity_results):
    print("Identifying queries containing rare features...")
    rare_query_ids = set()
    rare_tokens = rarity_results['rare_tokens']
    rare_trigram_hash_indices = rarity_results['rare_trigram_hash_indices']
    trigram_hasher = rarity_results['trigram_hasher']
    if not rare_tokens and not rare_trigram_hash_indices:
         print("Warning: No rare tokens or trigram hashes identified. Cannot find rare feature queries.")
         return []
    def tokenize(text): return preprocess_text_simple(text).split()
    for query_id, query_text in tqdm(queries_dict.items(), desc="Checking Queries for Rarity"):
        query_tokens = set(tokenize(query_text))
        if query_tokens.intersection(rare_tokens):
            rare_query_ids.add(query_id)
            continue
        if rare_trigram_hash_indices and trigram_hasher:
            try:
                query_trigram_sparse = trigram_hasher.transform([preprocess_text_simple(query_text)])
                query_hash_indices = set(query_trigram_sparse.indices)
                if query_hash_indices.intersection(rare_trigram_hash_indices):
                    rare_query_ids.add(query_id)
            except Exception as e: print(f"Warning: Could not process query '{query_id}' for trigram check: {e}")
    print(f"Found {len(rare_query_ids)} queries containing potentially rare features.")
    return list(rare_query_ids)

def evaluate_runs(results_dict, qrels_dict, metrics, method_name):
    if not IR_MEASURES_AVAILABLE:
        print(f"Skipping evaluation for {method_name}: ir_measures not available.")
        return None
    print(f"\nCalculating benchmark metrics for {method_name}...")
    evaluator = ir_measures.evaluator(metrics, qrels_dict)
    all_metrics = {}
    for run_name, run_data in results_dict.items():
         print(f" Evaluating: {run_name}")
         if not run_data:
             print(f"  Skipping {run_name} - no results.")
             metrics_agg = {str(m): 0.0 for m in metrics}
         else:
             metrics_agg = evaluator.evaluate(run_data)
         all_metrics[run_name] = metrics_agg
    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics.columns = [str(m) for m in metrics]
    df_metrics.index.name = "Method"
    print(df_metrics.round(4))
    return df_metrics

# --- NEW Batched Search Function ---
def run_batched_search(index, query_embeddings, k, doc_ids_map, is_l2=False):
    """
    Performs batched search on a Faiss index.

    Args:
        index: The initialized Faiss index.
        query_embeddings: A NumPy array of query embeddings (num_queries, dim).
        k: Number of neighbors to retrieve.
        doc_ids_map: List or array mapping Faiss index positions to actual doc IDs.
        is_l2: Boolean, set True if using L2 distance (requires score inversion).

    Returns:
        A list of lists, where each inner list contains tuples of (doc_id, score)
        for the corresponding query. Returns None if input is invalid.
    """
    if index is None or query_embeddings is None or query_embeddings.shape[0] == 0:
        return None

    num_queries = query_embeddings.shape[0]
    all_results = [[] for _ in range(num_queries)] # Pre-allocate result lists

    # Ensure index is not None before using ntotal
    if index.ntotal == 0:
        print("Warning: Attempting to search an empty index.")
        return all_results # Return empty results for all queries

    # Faiss search requires float32
    query_embeddings = query_embeddings.astype('float32')

    try:
        # Perform the batched search
        distances, indices = index.search(query_embeddings, k)

        # Process results
        for i in range(num_queries):
            for j in range(k):
                idx = indices[i, j]
                dist = distances[i, j]

                # Check for invalid index (-1 usually means fewer than k results found)
                # Also check against index size just in case
                if idx == -1 or idx >= len(doc_ids_map):
                    continue

                doc_id = doc_ids_map[idx]
                score = 1.0 / (1.0 + dist) if is_l2 else dist # Invert L2 distance, keep IP/Cosine as is

                all_results[i].append((doc_id, score))

    except Exception as e:
        print(f"Error during Faiss batched search: {e}")
        # Return partially filled or empty results depending on where the error occurred
        return all_results

    return all_results


# --- Experiment 1: MS MARCO ---

def run_msmarco_experiment(output_dir_exp1):
    """Runs ANN limitation experiment using MS MARCO dataset with batched search."""
    print("\n" + "="*20 + " Running Experiment 1: MS MARCO (Batched Search) " + "="*20)
    os.makedirs(output_dir_exp1, exist_ok=True)

    # --- 1. Load Data (MS MARCO Passage Ranking) ---
    # ... (Keep data loading exactly the same) ...
    print("Loading MS MARCO dataset (passage ranking)...")
    try:
        corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
        corpus = corpus.rename_columns({"_id": "doc_id", "text": "passage_text"})
        corpus = corpus.map(lambda x: {'doc_id': str(x['doc_id'])})
        if MSMARCO_CORPUS_LIMIT is not None and MSMARCO_CORPUS_LIMIT < len(corpus):
            print(f"Limiting corpus to {MSMARCO_CORPUS_LIMIT} passages.")
            corpus = corpus.select(range(MSMARCO_CORPUS_LIMIT))
            gc.collect()
        corpus_ids = set(corpus['doc_id'])
        print(f"Loaded {len(corpus)} passages.")
        queries_dev = load_dataset('BeIR/msmarco', 'queries', split='dev')
        queries_dev = queries_dev.rename_columns({"_id": "query_id", "text": "query_text"})
        queries_dev_dict = {q['query_id']: q['query_text'] for q in queries_dev}
        print(f"Loaded {len(queries_dev_dict)} dev queries.")
        qrels_dev = load_dataset('BeIR/msmarco-qrels', split='dev')
        print("Filtering qrels to match loaded corpus and queries...")
        qrels_dict = collections.defaultdict(dict)
        valid_qrels_count = 0; query_ids_in_qrels = set()
        for qrel in qrels_dev:
            doc_id = str(qrel['corpus-id']); query_id = qrel['query-id']
            if query_id in queries_dev_dict and doc_id in corpus_ids:
                qrels_dict[query_id][doc_id] = 1; valid_qrels_count += 1; query_ids_in_qrels.add(query_id)
        print(f"Loaded {valid_qrels_count} relevant qrels pairs matching the data.")
        queries_dev_dict_filtered = {qid: text for qid, text in queries_dev_dict.items() if qid in query_ids_in_qrels}
        print(f"Filtered to {len(queries_dev_dict_filtered)} dev queries with relevant docs in corpus.")
        corpus_texts_msmarco = [preprocess_text_simple(t) for t in corpus['passage_text']]
        doc_ids_msmarco = corpus['doc_id']
    except Exception as e:
        print(f"Error loading MS MARCO data: {e}"); return

    # --- 2. Rarity Analysis (on Corpus) ---
    # ... (Keep rarity analysis exactly the same) ...
    rarity_cache_file = os.path.join(output_dir_exp1, 'msmarco_rarity_analysis.pkl')
    rarity_results = analyze_rarity(corpus_texts_msmarco, rarity_cache_file, token_max_df=RARE_TOKEN_MAX_DF, trigram_max_freq=RARE_TRIGRAM_MAX_FREQ)

    # --- 3. Find Rare Feature Queries (from Dev Set) ---
    # ... (Keep query finding exactly the same) ...
    rare_query_ids = find_rare_feature_queries(queries_dev_dict_filtered, rarity_results)
    if NUM_RARE_QUERY_SAMPLES and len(rare_query_ids) > NUM_RARE_QUERY_SAMPLES:
         print(f"Sampling {NUM_RARE_QUERY_SAMPLES} rare queries for evaluation.")
         rare_query_ids = random.sample(rare_query_ids, NUM_RARE_QUERY_SAMPLES)
    qrels_dict_rare = {qid: docs for qid, docs in qrels_dict.items() if qid in rare_query_ids}
    print(f"Created rare query subset with {len(rare_query_ids)} queries and {len(qrels_dict_rare)} qrels entries.")

    # --- 4. Embedding Methods & Indexing ---
    # ... (Keep SBERT and TrigramRP embedding/indexing logic exactly the same) ...
    # === 4.a SBERT Embedding ===
    print("\n--- Processing SBERT Embeddings ---"); sbert_output_dir = os.path.join(output_dir_exp1, 'sbert'); os.makedirs(sbert_output_dir, exist_ok=True)
    sbert_emb_file = os.path.join(sbert_output_dir, f'corpus_embeddings_{MSMARCO_CORPUS_LIMIT}.npy'); sbert_ids_file = os.path.join(sbert_output_dir, f'doc_ids_{MSMARCO_CORPUS_LIMIT}.npy')
    sbert_model = None; corpus_embeddings_sbert = None
    # (Caching logic...)
    if os.path.exists(sbert_emb_file) and os.path.exists(sbert_ids_file):
        print("Loading cached SBERT embeddings..."); corpus_embeddings_sbert = np.load(sbert_emb_file); loaded_ids = np.load(sbert_ids_file)
        if len(loaded_ids) == len(doc_ids_msmarco) and corpus_embeddings_sbert.shape[0] == len(doc_ids_msmarco): print("Using cached SBERT embeddings.")
        else: print("Cache mismatch. Re-embedding SBERT..."); corpus_embeddings_sbert = None
    else: corpus_embeddings_sbert = None
    if corpus_embeddings_sbert is None:
        print(f"Loading SBERT model: {SBERT_MODEL_NAME}"); sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        print("Embedding corpus with SBERT (normalize=True)..."); corpus_embeddings_sbert = sbert_model.encode( corpus_texts_msmarco, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        np.save(sbert_emb_file, corpus_embeddings_sbert); np.save(sbert_ids_file, np.array(doc_ids_msmarco))
    else: print(f"Loading SBERT model: {SBERT_MODEL_NAME}"); sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    # SBERT Index Building (IP / Cosine)
    print("Building Faiss indices for SBERT (IP)..."); EMBEDDING_DIM_SBERT = corpus_embeddings_sbert.shape[1] if corpus_embeddings_sbert is not None else 0
    index_flat_sbert = faiss.IndexFlatIP(EMBEDDING_DIM_SBERT) if EMBEDDING_DIM_SBERT > 0 else None
    index_ivf_sbert = None; index_hnsw_sbert = None
    if index_flat_sbert and corpus_embeddings_sbert is not None and index_flat_sbert.d == corpus_embeddings_sbert.shape[1]:
        index_flat_sbert.add(corpus_embeddings_sbert); quantizer_ip = faiss.IndexFlatIP(EMBEDDING_DIM_SBERT)
        index_ivf_sbert = faiss.IndexIVFFlat(quantizer_ip, EMBEDDING_DIM_SBERT, NLIST_IVF, faiss.METRIC_INNER_PRODUCT); index_ivf_sbert.train(corpus_embeddings_sbert); index_ivf_sbert.add(corpus_embeddings_sbert); index_ivf_sbert.nprobe = NPROBE_IVF
        index_hnsw_sbert = faiss.IndexHNSWFlat(EMBEDDING_DIM_SBERT, HNSW_M, faiss.METRIC_INNER_PRODUCT); index_hnsw_sbert.hnsw.efConstruction = EF_CONSTRUCTION_HNSW; index_hnsw_sbert.hnsw.efSearch = EF_SEARCH_HNSW; index_hnsw_sbert.add(corpus_embeddings_sbert); print("SBERT Indices built.")
    else: print("Warning: SBERT setup failed or dimension mismatch, skipping index building.")

    # === 4.b Trigram+RP Embedding ===
    print("\n--- Processing Trigram+RP Embeddings ---"); trigramrp_output_dir = os.path.join(output_dir_exp1, 'trigram_rp'); os.makedirs(trigramrp_output_dir, exist_ok=True)
    trigramrp_emb_file = os.path.join(trigramrp_output_dir, f'corpus_embeddings_{MSMARCO_CORPUS_LIMIT}.npy'); trigramrp_ids_file = os.path.join(trigramrp_output_dir, f'doc_ids_{MSMARCO_CORPUS_LIMIT}.npy'); trigramrp_pipe_file = os.path.join(trigramrp_output_dir, 'encoding_pipeline.pkl')
    corpus_embeddings_rp = None; encoding_pipeline_rp = None
    # (Caching logic...)
    if os.path.exists(trigramrp_emb_file) and os.path.exists(trigramrp_ids_file) and os.path.exists(trigramrp_pipe_file):
        print("Loading cached Trigram+RP embeddings and pipeline..."); corpus_embeddings_rp = np.load(trigramrp_emb_file); loaded_ids = np.load(trigramrp_ids_file);
        with open(trigramrp_pipe_file, 'rb') as f: encoding_pipeline_rp = pickle.load(f)
        if len(loaded_ids) == len(doc_ids_msmarco) and corpus_embeddings_rp.shape[0] == len(doc_ids_msmarco): print("Using cached Trigram+RP embeddings and pipeline.")
        else: print("Cache mismatch. Re-embedding Trigram+RP..."); corpus_embeddings_rp = None; encoding_pipeline_rp = None
    else: corpus_embeddings_rp = None; encoding_pipeline_rp = None
    if corpus_embeddings_rp is None:
        print("Defining and fitting Trigram+RP pipeline..."); hasher = HashingVectorizer(n_features=N_FEATURES_SPARSE, analyzer='char', ngram_range=(3, 3), norm=None, alternate_sign=False); projector = GaussianRandomProjection(n_components=N_COMPONENTS_DENSE, random_state=42); encoding_pipeline_rp = Pipeline([('hasher', hasher), ('projector', projector)])
        print("Embedding corpus with Trigram+RP..."); corpus_embeddings_rp = encoding_pipeline_rp.fit_transform(corpus_texts_msmarco); corpus_embeddings_rp = corpus_embeddings_rp.astype('float32')
        print("Saving Trigram+RP embeddings and pipeline..."); np.save(trigramrp_emb_file, corpus_embeddings_rp); np.save(trigramrp_ids_file, np.array(doc_ids_msmarco));
        with open(trigramrp_pipe_file, 'wb') as f: pickle.dump(encoding_pipeline_rp, f)
    elif encoding_pipeline_rp is None: print("Error: Pipeline file missing despite embeddings existing."); return
    # Trigram+RP Index Building (L2)
    print("Building Faiss indices for Trigram+RP (L2)..."); EMBEDDING_DIM_RP = corpus_embeddings_rp.shape[1] if corpus_embeddings_rp is not None else 0
    index_flat_rp = faiss.IndexFlatL2(EMBEDDING_DIM_RP) if EMBEDDING_DIM_RP > 0 else None
    index_ivf_rp = None; index_hnsw_rp = None
    if index_flat_rp and corpus_embeddings_rp is not None and index_flat_rp.d == corpus_embeddings_rp.shape[1]:
        index_flat_rp.add(corpus_embeddings_rp); quantizer_l2 = faiss.IndexFlatL2(EMBEDDING_DIM_RP)
        index_ivf_rp = faiss.IndexIVFFlat(quantizer_l2, EMBEDDING_DIM_RP, NLIST_IVF, faiss.METRIC_L2); index_ivf_rp.train(corpus_embeddings_rp); index_ivf_rp.add(corpus_embeddings_rp); index_ivf_rp.nprobe = NPROBE_IVF
        index_hnsw_rp = faiss.IndexHNSWFlat(EMBEDDING_DIM_RP, HNSW_M, faiss.METRIC_L2); index_hnsw_rp.hnsw.efConstruction = EF_CONSTRUCTION_HNSW; index_hnsw_rp.hnsw.efSearch = EF_SEARCH_HNSW; index_hnsw_rp.add(corpus_embeddings_rp); print("Trigram+RP Indices built.")
    else: print("Warning: Trigram+RP setup failed or dimension mismatch, skipping index building.")


    # --- 5. Evaluation Runs (Batched) ---

    # Function to process benchmark queries in batches
    def process_benchmark_batch(query_ids_batch, query_texts_batch, encoder, index_flat, index_ivf, index_hnsw, doc_ids_map, k, is_l2, is_sbert=False):
        results_enn, results_ivf, results_hnsw = [], [], []
        if not query_ids_batch: return results_enn, results_ivf, results_hnsw

        # Encode batch
        texts_to_encode = [preprocess_text_simple(t) for t in query_texts_batch]
        if is_sbert:
            query_embs = encoder.encode(texts_to_encode, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        else: # TrigramRP Pipeline
            query_embs = encoder.transform(texts_to_encode).astype('float32')

        # Search batch
        enn_batch_results = run_batched_search(index_flat, query_embs, k, doc_ids_map, is_l2)
        ivf_batch_results = run_batched_search(index_ivf, query_embs, k, doc_ids_map, is_l2)
        hnsw_batch_results = run_batched_search(index_hnsw, query_embs, k, doc_ids_map, is_l2)

        # Format results for ir_measures (List of Tuples)
        for i, qid in enumerate(query_ids_batch):
            if enn_batch_results and enn_batch_results[i]: results_enn.extend([(qid, did, score) for did, score in enn_batch_results[i]])
            if ivf_batch_results and ivf_batch_results[i]: results_ivf.extend([(qid, did, score) for did, score in ivf_batch_results[i]])
            if hnsw_batch_results and hnsw_batch_results[i]: results_hnsw.extend([(qid, did, score) for did, score in hnsw_batch_results[i]])

        return results_enn, results_ivf, results_hnsw

    # === 5.a Overall Benchmark Evaluation ===
    print("\n--- Running Overall Benchmark Evaluation (MS MARCO Dev - Batched) ---")
    benchmark_metrics = [RR(rel=1)@10, Recall(rel=1)@100, Recall(rel=1)@1000, nDCG@10]
    all_benchmark_runs = collections.defaultdict(list) # Use defaultdict

    # Prepare query list for batching
    query_items = list(queries_dev_dict_filtered.items())
    num_batches = math.ceil(len(query_items) / SEARCH_BATCH_SIZE)

    # SBERT Runs (Batched)
    if sbert_model and index_flat_sbert and index_ivf_sbert and index_hnsw_sbert:
        print(" Running SBERT Benchmark Queries (Batched)...")
        for i in tqdm(range(num_batches), desc="SBERT Batches"):
            batch_start = i * SEARCH_BATCH_SIZE
            batch_end = batch_start + SEARCH_BATCH_SIZE
            batch_items = query_items[batch_start:batch_end]
            batch_qids = [item[0] for item in batch_items]
            batch_qtexts = [item[1] for item in batch_items]

            enn_res, ivf_res, hnsw_res = process_benchmark_batch(
                batch_qids, batch_qtexts, sbert_model, index_flat_sbert, index_ivf_sbert, index_hnsw_sbert,
                doc_ids_msmarco, K_NEIGHBORS_MSMARCO, is_l2=False, is_sbert=True
            )
            all_benchmark_runs['SBERT_ENN'].extend(enn_res)
            all_benchmark_runs['SBERT_IVF'].extend(ivf_res)
            all_benchmark_runs['SBERT_HNSW'].extend(hnsw_res)

    # Trigram+RP Runs (Batched)
    if encoding_pipeline_rp and index_flat_rp and index_ivf_rp and index_hnsw_rp:
        print(" Running Trigram+RP Benchmark Queries (Batched)...")
        for i in tqdm(range(num_batches), desc="TrigramRP Batches"):
            batch_start = i * SEARCH_BATCH_SIZE
            batch_end = batch_start + SEARCH_BATCH_SIZE
            batch_items = query_items[batch_start:batch_end]
            batch_qids = [item[0] for item in batch_items]
            batch_qtexts = [item[1] for item in batch_items]

            enn_res, ivf_res, hnsw_res = process_benchmark_batch(
                batch_qids, batch_qtexts, encoding_pipeline_rp, index_flat_rp, index_ivf_rp, index_hnsw_rp,
                doc_ids_msmarco, K_NEIGHBORS_MSMARCO, is_l2=True, is_sbert=False
            )
            all_benchmark_runs['TrigramRP_ENN'].extend(enn_res)
            all_benchmark_runs['TrigramRP_IVF'].extend(ivf_res)
            all_benchmark_runs['TrigramRP_HNSW'].extend(hnsw_res)


    # Evaluate Benchmark Runs
    benchmark_results_df = evaluate_runs(all_benchmark_runs, qrels_dict, benchmark_metrics, "Overall Benchmark")
    if benchmark_results_df is not None:
        # Verification Check for SBERT
        try:
            sbert_mrr = benchmark_results_df.loc['SBERT_ENN', str(RR(rel=1)@10)]
            print(f"\nSBERT ENN MRR@10 Verification:"); print(f"  Expected : ~{EXPECTED_SBERT_MRR:.4f}"); print(f"  Achieved : {sbert_mrr:.4f}")
            if abs(sbert_mrr - EXPECTED_SBERT_MRR) > 0.02: print("  WARNING: Achieved MRR deviates significantly from expected value.")
            else: print("  OK: Achieved MRR is close to expected value.")
        except KeyError: print(" Could not perform SBERT MRR verification (results missing).")
        benchmark_results_df.to_csv(os.path.join(output_dir_exp1, "benchmark_metrics_overall.csv"))


    # === 5.b Rare Feature Query Subset Evaluation ===
    print("\n--- Running Rare Feature Query Subset Evaluation (Batched) ---")
    if not rare_query_ids:
        print("Skipping evaluation: No rare feature queries identified.")
    else:
        all_rare_runs = collections.defaultdict(list) # Filter existing runs for rare query IDs
        for run_name, run_data in all_benchmark_runs.items():
             # Filter the runs generated above - more efficient than re-running search
             all_rare_runs[run_name] = [(qid, did, score) for qid, did, score in run_data if qid in rare_query_ids]

        rare_results_df = evaluate_runs(all_rare_runs, qrels_dict_rare, benchmark_metrics, "Rare Query Subset")
        if rare_results_df is not None:
             rare_results_df.to_csv(os.path.join(output_dir_exp1, "benchmark_metrics_rare_subset.csv"))

    # === 5.c Specific "ID Query" Test (Trigram+RP only - Batched) ===
    print("\n--- Running Specific ID Query Test (Trigram+RP - Batched) ---")
    id_query_results = []
    if encoding_pipeline_rp and index_flat_rp and index_ivf_rp and index_hnsw_rp:
        num_id_queries = 500 # Increase number now that it's batched
        ids_to_test = random.sample(doc_ids_msmarco, min(num_id_queries, len(doc_ids_msmarco)))
        id_query_texts = ["id_" + preprocess_text_simple(tid) for tid in ids_to_test]

        # Encode ID queries in one go
        id_query_embs = encoding_pipeline_rp.transform(id_query_texts).astype('float32')

        # Search using K=1 for Rank 1 check
        k_id = 1
        # ENN
        D_flat_id, I_flat_id = index_flat_rp.search(id_query_embs, k_id)
        # IVF
        D_ivf_id, I_ivf_id = index_ivf_rp.search(id_query_embs, k_id)
        # HNSW
        D_hnsw_id, I_hnsw_id = index_hnsw_rp.search(id_query_embs, k_id)

        # Process results
        for i, target_id in enumerate(tqdm(ids_to_test, desc="Processing ID Results")):
            res_row = {'Target ID': target_id}
            # ENN
            enn_idx = I_flat_id[i, 0]
            enn_found_id = doc_ids_msmarco[enn_idx] if enn_idx >= 0 else None
            res_row['ENN Rank1 Match'] = (enn_found_id == target_id)
            res_row['ENN Rank1 Dist'] = D_flat_id[i, 0] if enn_idx >=0 else np.inf
            # IVF
            ivf_idx = I_ivf_id[i, 0]
            ivf_found_id = doc_ids_msmarco[ivf_idx] if ivf_idx != -1 else None
            res_row['IVF Rank1 Match'] = (ivf_found_id == target_id)
            res_row['IVF Rank1 Dist'] = D_ivf_id[i, 0] if ivf_idx !=-1 else np.inf
            # HNSW
            hnsw_idx = I_hnsw_id[i, 0]
            hnsw_found_id = doc_ids_msmarco[hnsw_idx] if hnsw_idx != -1 else None
            res_row['HNSW Rank1 Match'] = (hnsw_found_id == target_id)
            res_row['HNSW Rank1 Dist'] = D_hnsw_id[i, 0] if hnsw_idx !=-1 else np.inf
            id_query_results.append(res_row)

        id_query_df = pd.DataFrame(id_query_results)
        print("\nID Query Test Summary (Trigram+RP - MS MARCO):")
        print(f" Total Queries: {len(id_query_df)}")
        print(f" ENN Rank1 Success Rate: {id_query_df['ENN Rank1 Match'].mean():.2%}")
        print(f" IVF Rank1 Success Rate: {id_query_df['IVF Rank1 Match'].mean():.2%}")
        print(f" HNSW Rank1 Success Rate: {id_query_df['HNSW Rank1 Match'].mean():.2%}")
        id_query_df.to_csv(os.path.join(output_dir_exp1, "id_query_test_trigramrp.csv"), index=False)
    else:
        print("Skipping ID Query Test: Trigram+RP setup not available.")

    # Clean up
    del corpus, corpus_texts_msmarco, corpus_embeddings_sbert, corpus_embeddings_rp
    del index_flat_sbert, index_ivf_sbert, index_hnsw_sbert
    del index_flat_rp, index_ivf_rp, index_hnsw_rp
    gc.collect()
    print("\n" + "="*20 + " Experiment 1 Finished " + "="*20)


# --- Experiment 2: Home Depot (Batched) ---

def run_home_depot_experiment(output_dir_exp2_base, trigram_mode):
    """Runs Home Depot experiment for a specific trigram mode with batched search."""
    output_dir_exp2 = os.path.join(output_dir_exp2_base, trigram_mode)
    print("\n" + "="*20 + f" Running Experiment 2: Home Depot (Mode: {trigram_mode} - Batched) " + "="*20)
    os.makedirs(output_dir_exp2, exist_ok=True)

    # --- File Checks & Load Data ---
    # ... (Keep data loading exactly the same, including text prep) ...
    if not os.path.exists(HOME_DEPOT_TRAIN_CSV): print(f"Error: Home Depot train.csv not found at {HOME_DEPOT_TRAIN_CSV}"); return
    print("Loading Home Depot data..."); df_train = pd.read_csv(HOME_DEPOT_TRAIN_CSV, encoding="ISO-8859-1")
    if os.path.exists(HOME_DEPOT_DESC_CSV): df_desc = pd.read_csv(HOME_DEPOT_DESC_CSV, encoding="ISO-8859-1"); df_train = pd.merge(df_train, df_desc, on='product_uid', how='left')
    df_train = df_train.fillna({'product_title': '', 'product_description': ''}); df_train['full_text'] = df_train['product_title'] + " " + df_train['product_description']; df_train['doc_id'] = df_train['product_uid'].astype(str)
    print(f"Preparing text for encoding (Mode: {trigram_mode})...")
    df_train['text_for_encoding'] = df_train.apply( lambda row: prepare_text_for_encoding_hd(row, trigram_mode, preprocess_text_hd), axis=1)
    if HD_CORPUS_SIZE_LIMIT: df_corpus = df_train.sample(n=min(HD_CORPUS_SIZE_LIMIT, len(df_train)), random_state=42).copy(); print(f"Using limited corpus size: {len(df_corpus)}")
    else: df_corpus = df_train.copy(); print(f"Using full corpus size: {len(df_corpus)}")
    doc_ids_hd = df_corpus['doc_id'].tolist(); corpus_texts_hd = df_corpus['text_for_encoding'].tolist()

    # --- Encoding Pipeline & Corpus Embedding ---
    # ... (Keep embedding logic exactly the same, including caching) ...
    hd_emb_file = os.path.join(output_dir_exp2, f'corpus_embeddings_{HD_CORPUS_SIZE_LIMIT}.npy'); hd_ids_file = os.path.join(output_dir_exp2, f'doc_ids_{HD_CORPUS_SIZE_LIMIT}.npy'); hd_pipe_file = os.path.join(output_dir_exp2, 'encoding_pipeline.pkl')
    corpus_embeddings_hd_rp = None; encoding_pipeline_hd_rp = None
    # (Caching logic...)
    if os.path.exists(hd_emb_file) and os.path.exists(hd_ids_file) and os.path.exists(hd_pipe_file):
        print("Loading cached HD Trigram+RP embeddings and pipeline..."); corpus_embeddings_hd_rp = np.load(hd_emb_file); loaded_ids = np.load(hd_ids_file)
        with open(hd_pipe_file, 'rb') as f: encoding_pipeline_hd_rp = pickle.load(f)
        if len(loaded_ids) == len(doc_ids_hd) and corpus_embeddings_hd_rp.shape[0] == len(doc_ids_hd): print("Using cached HD Trigram+RP embeddings and pipeline.")
        else: print("Cache mismatch. Re-embedding HD Trigram+RP..."); corpus_embeddings_hd_rp, encoding_pipeline_hd_rp = None, None
    else: corpus_embeddings_hd_rp, encoding_pipeline_hd_rp = None, None
    if corpus_embeddings_hd_rp is None:
        print("Defining and fitting Trigram+RP pipeline for HD..."); hasher = HashingVectorizer(n_features=N_FEATURES_SPARSE, analyzer='char', ngram_range=(3, 3), norm=None, alternate_sign=False); projector = GaussianRandomProjection(n_components=N_COMPONENTS_DENSE, random_state=42); encoding_pipeline_hd_rp = Pipeline([('hasher', hasher), ('projector', projector)])
        print("Embedding HD corpus with Trigram+RP..."); corpus_embeddings_hd_rp = encoding_pipeline_hd_rp.fit_transform(corpus_texts_hd); corpus_embeddings_hd_rp = corpus_embeddings_hd_rp.astype('float32')
        print("Saving HD Trigram+RP embeddings and pipeline..."); np.save(hd_emb_file, corpus_embeddings_hd_rp); np.save(hd_ids_file, np.array(doc_ids_hd));
        with open(hd_pipe_file, 'wb') as f: pickle.dump(encoding_pipeline_hd_rp, f)
    elif encoding_pipeline_hd_rp is None: print("Error: HD Pipeline file missing."); return
    # Query Encoding Function specific to HD pipeline and mode
    def encode_query_hd_rp_batch(query_texts_batch): # Batched version
        final_query_texts = []
        for query_text in query_texts_batch:
            is_id_q = query_text.startswith("id_")
            if is_id_q: processed_q_text = query_text
            else: processed_q_text = preprocess_text_hd(query_text)
            if trigram_mode == 'within_token':
                separator = " _TOKSEP_ "
                if is_id_q: final_query_text = processed_q_text
                else: final_query_text = separator.join(processed_q_text.split())
            else: # across_tokens
                 final_query_text = processed_q_text
            final_query_texts.append(final_query_text)
        return encoding_pipeline_hd_rp.transform(final_query_texts).astype('float32')

    # --- Build Faiss Indices (L2) ---
    # ... (Keep index building exactly the same) ...
    print("Building Faiss indices for HD Trigram+RP (L2)..."); EMBEDDING_DIM_HD_RP = corpus_embeddings_hd_rp.shape[1] if corpus_embeddings_hd_rp is not None else 0
    index_flat_hd_rp = faiss.IndexFlatL2(EMBEDDING_DIM_HD_RP) if EMBEDDING_DIM_HD_RP > 0 else None
    index_ivf_hd_rp = None; index_hnsw_hd_rp = None
    if index_flat_hd_rp and corpus_embeddings_hd_rp is not None and index_flat_hd_rp.d == corpus_embeddings_hd_rp.shape[1]:
        index_flat_hd_rp.add(corpus_embeddings_hd_rp); quantizer_l2 = faiss.IndexFlatL2(EMBEDDING_DIM_HD_RP)
        index_ivf_hd_rp = faiss.IndexIVFFlat(quantizer_l2, EMBEDDING_DIM_HD_RP, NLIST_IVF, faiss.METRIC_L2); index_ivf_hd_rp.train(corpus_embeddings_hd_rp); index_ivf_hd_rp.add(corpus_embeddings_hd_rp); index_ivf_hd_rp.nprobe = NPROBE_IVF
        index_hnsw_hd_rp = faiss.IndexHNSWFlat(EMBEDDING_DIM_HD_RP, HNSW_M, faiss.METRIC_L2); index_hnsw_hd_rp.hnsw.efConstruction = EF_CONSTRUCTION_HNSW; index_hnsw_hd_rp.hnsw.efSearch = EF_SEARCH_HNSW; index_hnsw_hd_rp.add(corpus_embeddings_hd_rp); print("HD Trigram+RP Indices built.")
    else: print("Warning: HD setup failed or dimension mismatch."); return


    # --- 5. Evaluation Runs (Batched) ---

    # === 5.a Overall Benchmark Evaluation (Kaggle Queries - Batched) ===
    print("\n--- Running Overall Benchmark Evaluation (Home Depot - Batched) ---")
    if not IR_MEASURES_AVAILABLE:
        print("Skipping benchmark: ir_measures not available.")
    else:
        print("Loading HD benchmark qrels..."); qrels_df = pd.read_csv(HOME_DEPOT_TRAIN_CSV, encoding="ISO-8859-1")
        qrels_df['query_id'] = qrels_df['search_term']; qrels_df['doc_id'] = qrels_df['product_uid'].astype(str); qrels_df['relevance'] = qrels_df['relevance'].astype(int)
        qrels_hd = collections.defaultdict(dict);
        for row in qrels_df.itertuples(): qrels_hd[row.query_id][row.doc_id] = row.relevance
        indexed_doc_ids_set_hd = set(doc_ids_hd); filtered_qrels_hd = collections.defaultdict(dict); valid_queries_hd = set()
        for qid, doc_rels in qrels_hd.items():
            filtered_docs = {did: rel for did, rel in doc_rels.items() if did in indexed_doc_ids_set_hd}
            if filtered_docs: filtered_qrels_hd[qid] = filtered_docs; valid_queries_hd.add(qid)
        print(f"Filtered HD qrels to {len(valid_queries_hd)} queries with relevant docs in index.")

        hd_benchmark_runs = collections.defaultdict(list)
        if filtered_qrels_hd:
            print("Running HD benchmark queries (Batched)...")
            benchmark_query_items_hd = qrels_df[qrels_df['query_id'].isin(valid_queries_hd)][['query_id', 'search_term']].drop_duplicates().values.tolist()
            num_batches_hd = math.ceil(len(benchmark_query_items_hd) / SEARCH_BATCH_SIZE)

            for i in tqdm(range(num_batches_hd), desc=f"HD Benchmark ({trigram_mode})"):
                batch_start = i * SEARCH_BATCH_SIZE
                batch_end = batch_start + SEARCH_BATCH_SIZE
                batch_items = benchmark_query_items_hd[batch_start:batch_end]
                batch_qids = [item[0] for item in batch_items]
                batch_qtexts = [item[1] for item in batch_items]

                # Encode batch using HD specific function
                q_emb_batch = encode_query_hd_rp_batch(batch_qtexts)

                # Search batch
                enn_batch_results = run_batched_search(index_flat_hd_rp, q_emb_batch, K_NEIGHBORS_HD, doc_ids_hd, is_l2=True)
                ivf_batch_results = run_batched_search(index_ivf_hd_rp, q_emb_batch, K_NEIGHBORS_HD, doc_ids_hd, is_l2=True)
                hnsw_batch_results = run_batched_search(index_hnsw_hd_rp, q_emb_batch, K_NEIGHBORS_HD, doc_ids_hd, is_l2=True)

                # Format results
                for batch_idx, qid in enumerate(batch_qids):
                     if enn_batch_results and enn_batch_results[batch_idx]: hd_benchmark_runs['HD_TrigramRP_ENN'].extend([(qid, did, score) for did, score in enn_batch_results[batch_idx]])
                     if ivf_batch_results and ivf_batch_results[batch_idx]: hd_benchmark_runs['HD_TrigramRP_IVF'].extend([(qid, did, score) for did, score in ivf_batch_results[batch_idx]])
                     if hnsw_batch_results and hnsw_batch_results[batch_idx]: hd_benchmark_runs['HD_TrigramRP_HNSW'].extend([(qid, did, score) for did, score in hnsw_batch_results[batch_idx]])

            # Evaluate
            hd_benchmark_metrics = [nDCG@10, Recall@10, P@10]
            hd_results_df = evaluate_runs(hd_benchmark_runs, filtered_qrels_hd, hd_benchmark_metrics, f"Home Depot Benchmark ({trigram_mode})")
            if hd_results_df is not None:
                hd_results_df.to_csv(os.path.join(output_dir_exp2, "benchmark_metrics_overall.csv"))
        else:
            print("No valid queries/qrels for HD benchmark evaluation.")


    # === 5.b Specific "ID Query" Test (Batched) ===
    print("\n--- Running Specific ID Query Test (Home Depot - Batched) ---")
    hd_id_query_results = []
    num_id_queries_hd = 500 # Increase sample size
    ids_to_test_hd = random.sample(doc_ids_hd, min(num_id_queries_hd, len(doc_ids_hd)))
    hd_id_query_texts = ["id_" + preprocess_text_hd(tid) for tid in ids_to_test_hd]

    # Encode batch
    hd_id_query_embs = encode_query_hd_rp_batch(hd_id_query_texts)

    # Search (K=1)
    k_id = 1
    D_flat_id, I_flat_id = index_flat_hd_rp.search(hd_id_query_embs, k_id)
    D_ivf_id, I_ivf_id = index_ivf_hd_rp.search(hd_id_query_embs, k_id)
    D_hnsw_id, I_hnsw_id = index_hnsw_hd_rp.search(hd_id_query_embs, k_id)

    # Process results
    for i, target_id in enumerate(tqdm(ids_to_test_hd, desc=f"Processing HD ID Results ({trigram_mode})")):
        res_row = {'Target ID': target_id}
        # ENN
        enn_idx = I_flat_id[i, 0]; enn_found_id = doc_ids_hd[enn_idx] if enn_idx >= 0 else None
        res_row['ENN Rank1 Match'] = (enn_found_id == target_id)
        # IVF
        ivf_idx = I_ivf_id[i, 0]; ivf_found_id = doc_ids_hd[ivf_idx] if ivf_idx != -1 else None
        res_row['IVF Rank1 Match'] = (ivf_found_id == target_id)
        # HNSW
        hnsw_idx = I_hnsw_id[i, 0]; hnsw_found_id = doc_ids_hd[hnsw_idx] if hnsw_idx != -1 else None
        res_row['HNSW Rank1 Match'] = (hnsw_found_id == target_id)
        hd_id_query_results.append(res_row)

    hd_id_query_df = pd.DataFrame(hd_id_query_results)
    print(f"\nID Query Test Summary (Home Depot - {trigram_mode}):")
    print(f" ENN Rank1 Success Rate: {hd_id_query_df['ENN Rank1 Match'].mean():.2%}")
    print(f" IVF Rank1 Success Rate: {hd_id_query_df['IVF Rank1 Match'].mean():.2%}")
    print(f" HNSW Rank1 Success Rate: {hd_id_query_df['HNSW Rank1 Match'].mean():.2%}")
    hd_id_query_df.to_csv(os.path.join(output_dir_exp2, "id_query_test.csv"), index=False)

    # Clean up
    del df_train, df_corpus, corpus_texts_hd, corpus_embeddings_hd_rp
    del index_flat_hd_rp, index_ivf_hd_rp, index_hnsw_hd_rp
    gc.collect()
    print("\n" + "="*20 + f" Experiment 2 Finished (Mode: {trigram_mode}) " + "="*20)


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting ANN Limitation Experiments (Batched Search)...")

    # Define output directories
    output_dir_exp1 = os.path.join(BASE_OUTPUT_DIR, 'experiment_1_msmarco')
    output_dir_exp2_base = os.path.join(BASE_OUTPUT_DIR, 'experiment_2_home_depot')

    # --- Run Experiment 1 (MS MARCO) ---
    try:
        run_msmarco_experiment(output_dir_exp1)
    except Exception as e:
        print(f"\n!!! Error during Experiment 1 (MS MARCO): {e} !!!")
        import traceback
        traceback.print_exc()

    # --- Run Experiment 2 (Home Depot - Both Modes) ---
    # Across Tokens Mode
    try:
        run_home_depot_experiment(output_dir_exp2_base, trigram_mode='across_tokens')
    except Exception as e:
        print(f"\n!!! Error during Experiment 2 (Home Depot - across_tokens): {e} !!!")
        import traceback
        traceback.print_exc()

    # Within Token Mode
    try:
        run_home_depot_experiment(output_dir_exp2_base, trigram_mode='within_token')
    except Exception as e:
        print(f"\n!!! Error during Experiment 2 (Home Depot - within_token): {e} !!!")
        import traceback
        traceback.print_exc()


    print("\nAll experiments finished. Check the output directories:")
    print(f" Experiment 1: {output_dir_exp1}")
    print(f" Experiment 2: {output_dir_exp2_base} (contains subfolders for 'across_tokens' and 'within_token')")