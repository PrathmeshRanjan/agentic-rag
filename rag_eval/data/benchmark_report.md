# RAG evaluation benchmark report

> Production-quality embedding run

---

## Retrieval benchmark ranking

Ranked by MRR@5, then Evidence Recall@5, Hit@5, and p95 latency.

| Rank | Configuration | Chunks | Reranker | Hit@1 | Hit@3 | Hit@5 | Evidence Recall@5 | Full Recall@5 | MRR@5 | nDCG@5 | p95 latency |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `recursive_512_0__rerank` | 80 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.950 | 1.000 | 1.000 | 0.956 | 0.900 | 0.975 | 0.946 | 200.8 ms |
| 2 | `semantic_90__rerank` | 52 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.950 | 1.000 | 1.000 | 0.944 | 0.875 | 0.975 | 0.940 | 310.6 ms |
| 3 | `fixed_512_100__rerank` | 81 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.925 | 1.000 | 1.000 | 0.944 | 0.875 | 0.963 | 0.924 | 166.2 ms |
| 4 | `recursive_1024_200__rerank` | 43 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.900 | 1.000 | 1.000 | 0.944 | 0.875 | 0.950 | 0.921 | 220.3 ms |
| 5 | `recursive_512_100__rerank` | 86 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.900 | 1.000 | 1.000 | 0.931 | 0.850 | 0.946 | 0.905 | 137.6 ms |
| 6 | `recursive_1024_0__rerank` | 41 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.875 | 1.000 | 1.000 | 0.944 | 0.875 | 0.938 | 0.911 | 226.2 ms |
| 7 | `fixed_512_0__rerank` | 72 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.900 | 0.975 | 0.975 | 0.944 | 0.900 | 0.933 | 0.917 | 149.1 ms |
| 8 | `recursive_512_0` | 80 | None | 0.850 | 0.950 | 0.975 | 0.925 | 0.875 | 0.898 | 0.879 | 35.9 ms |
| 9 | `fixed_256_64__rerank` | 167 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.775 | 0.925 | 0.950 | 0.906 | 0.850 | 0.852 | 0.840 | 106.1 ms |
| 10 | `recursive_1024_0` | 41 | None | 0.775 | 0.900 | 0.950 | 0.875 | 0.800 | 0.845 | 0.817 | 12.8 ms |
| 11 | `recursive_512_100` | 86 | None | 0.700 | 0.950 | 0.975 | 0.919 | 0.850 | 0.819 | 0.813 | 14.4 ms |
| 12 | `fixed_512_0` | 72 | None | 0.675 | 0.925 | 0.975 | 0.925 | 0.875 | 0.803 | 0.806 | 18.9 ms |
| 13 | `fixed_256_0__rerank` | 134 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.700 | 0.875 | 0.900 | 0.863 | 0.825 | 0.790 | 0.790 | 122.1 ms |
| 14 | `recursive_1024_200` | 43 | None | 0.650 | 0.925 | 0.950 | 0.875 | 0.800 | 0.784 | 0.772 | 13.3 ms |
| 15 | `fixed_256_64` | 167 | None | 0.625 | 0.875 | 0.925 | 0.875 | 0.825 | 0.753 | 0.761 | 16.6 ms |
| 16 | `fixed_512_100` | 81 | None | 0.600 | 0.900 | 0.975 | 0.912 | 0.850 | 0.747 | 0.763 | 33.9 ms |
| 17 | `semantic_90` | 52 | None | 0.600 | 0.900 | 0.950 | 0.887 | 0.825 | 0.728 | 0.744 | 13.9 ms |
| 18 | `fixed_256_0` | 134 | None | 0.550 | 0.800 | 0.875 | 0.825 | 0.775 | 0.680 | 0.692 | 40.0 ms |

---

## Analysis

- Best rank quality: `recursive_512_0__rerank` with MRR@5 **0.975** and Hit@5 **1.000**.
- Lowest p95 query latency: `recursive_1024_0` at **12.8 ms**.
- Across paired candidate sets, reranking changed Hit@3 by **+7.2 points**, MRR@5 by **+0.141**, and p95 latency by **+159.9 ms** on average.
- Failed query/config evaluations: **0**.

---

## Reproducibility and metric definitions

- Run ID: `e4a38d94-f1f1-4c86-80e6-8f6ddd5e35dc`; dataset SHA-256: `1e452e61df388d7c3f6b1b372382493fa8c8dc3161212ca8d68b8b328fabab37`.
- Backend: `chroma`; embeddings: `sentence-transformers:sentence-transformers/all-MiniLM-L6-v2`; candidate K: 20; evaluated K: 10.
- Hit@K asks whether any gold passage is present. Evidence Recall@K counts distinct required passages, so duplicate overlapping chunks receive no extra credit.
- MRR uses the first relevant rank; Full Recall@5 requires every passage for a compound question; latency reports end-to-end retrieval plus reranking.
- The reranker sees the exact cached first-stage candidates used by its paired baseline; model-load time is recorded separately from query latency.

---

## Key findings

- Evaluated **18 retrieval configurations** across **3 chunking strategies** on **40 grounded questions** from **3 documents**.
- Best-ranked configuration: `recursive_512_0__rerank`, achieving **100.0% Hit@5**, **0.975 MRR@5**, and **95.6% evidence Recall@5**.
- Mean cross-encoder impact on identical first-stage candidates: **+7.2 percentage points Hit@3** and **+0.141 MRR@5**, at **+159.9 ms p95 latency**.
