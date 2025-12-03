import numpy as np
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

class VectorStore:
    def __init__(self, model_name: str):
        # existing stuff
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.index = None          # FAISS index
        self.chunks = []           # list of dicts

        # NEW: BM25 state
        self.bm25 = None
        self._bm25_tokenized = None

    def load(self, path: str):
        """
        Load FAISS index and chunks from disk.
        """
        import os
        import pickle
        
        # Load FAISS index
        index_file = os.path.join(path, "index.faiss")
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            print(f"Loaded FAISS index from {index_file}")
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_file}")
        
        # Load chunks from pickle file
        chunks_file = f"{path}_chunks.pkl"
        if os.path.exists(chunks_file):
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
        else:
            raise FileNotFoundError(f"Chunks file not found at {chunks_file}")

        # Build BM25 after chunks are loaded
        self._build_bm25()

    # ---------- BM25 building ----------

    def _build_bm25(self):
        """
        Build BM25 index over the text of each chunk.
        """
        docs = []
        for c in self.chunks:
            text = c.get("text") or c.get("content") or ""
            docs.append(text)

        tokenized = [d.split() for d in docs]
        self._bm25_tokenized = tokenized
        self.bm25 = BM25Okapi(tokenized)
        print("BM25 index built over", len(docs), "chunks")

    @staticmethod
    def _min_max_norm(score_dict):
        """
        Normalize scores to [0, 1].
        score_dict: {id: score}
        """
        if not score_dict:
            return {}
        vals = np.array(list(score_dict.values()), dtype=float)
        mn, mx = vals.min(), vals.max()
        if mx == mn:
            # all same -> zero out
            return {k: 0.0 for k in score_dict.keys()}
        return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

    # ---------- Optional: keep your original vector-only search ----------
    
    def search(self, query: str, k: int = 5):
        """
        Default search method - calls search_vector for compatibility.
        """
        return self.search_vector(query, k)

    def search_vector(self, query: str, k: int = 5):
        """
        Pure vector / FAISS retrieval.
        If you already have `search()` doing this, you can skip this and keep that.
        """
        # Embed query using existing HuggingFaceEmbeddings
        q_vec = self.embeddings.embed_query(query)
        q_vec = np.array([q_vec], dtype="float32")  # (1, d)

        scores, indices = self.index.search(q_vec, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            chunk = self.chunks[int(idx)].copy()
            results.append({
                'chunk': chunk,
                'score': float(score),
                'rank': i + 1
            })
        return results

    # ---------- NEW: hybrid search (BM25 + Vector) ----------

    def search_hybrid(
        self,
        query: str,
        k: int = 5,
        top_k_vec: int = 20,
        top_k_bm25: int = 20,
        alpha: float = 0.6,  # weight for BM25
    ):
        """
        Hybrid retrieval: BM25 + FAISS vector search.
        Returns list of chunks with 'relevance_score' = hybrid score.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call _build_bm25() after loading chunks.")

        # --- 1) BM25 side ---
        tokens = query.split()
        bm25_scores_all = self.bm25.get_scores(tokens)  # shape: (N,)

        bm25_top_ids = np.argsort(-bm25_scores_all)[:top_k_bm25]
        bm25_dict = {int(i): float(bm25_scores_all[i]) for i in bm25_top_ids}

        # --- 2) Vector / FAISS side ---
        q_vec = self.embeddings.embed_query(query)
        q_vec = np.array([q_vec], dtype="float32")

        vec_scores, vec_ids = self.index.search(q_vec, top_k_vec)
        vec_scores = vec_scores[0]
        vec_ids = vec_ids[0]

        vec_dict = {int(idx): float(score) for idx, score in zip(vec_ids, vec_scores)}

        # --- 3) Normalize both ---
        bm25_norm = self._min_max_norm(bm25_dict)
        vec_norm = self._min_max_norm(vec_dict)

        # --- 4) Combine scores ---
        all_ids = set(bm25_norm.keys()) | set(vec_norm.keys())
        final_scores = {}
        for i in all_ids:
            b = bm25_norm.get(i, 0.0)
            v = vec_norm.get(i, 0.0)
            final_scores[i] = alpha * b + (1.0 - alpha) * v

        # --- 5) Sort and build result list ---
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for i, (doc_id, score) in enumerate(ranked):
            chunk = self.chunks[doc_id].copy()
            results.append({
                'chunk': chunk,
                'score': float(score),
                'rank': i + 1
            })

        return results
