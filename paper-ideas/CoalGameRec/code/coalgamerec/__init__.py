"""CoalGameRec local executable prototype.

This package implements the end-to-end empirical case-study pipeline described in
paper-ideas/CoalGameRec. It is designed to run on a Mac M4 Pro class laptop for
feasibility/prototyping. HCCF official-port validation remains a real artifact;
this code provides a self-contained LightGCN-style backbone and the complete
post-hoc attribution/reranking pipeline.
"""

__all__ = ["data", "metrics", "models", "attribution", "rerank", "stats"]
