"""FAISS-based retriever for the NaturalProofs dataset.

Provides semantic search over mathematical theorems and definitions
using sentence embeddings and FAISS for efficient similarity search.
"""

import json
import logging
import os
from typing import Iterator

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever, SearchResult
from ..config import RetrieverConfig, EmbeddingConfig

logger = logging.getLogger(__name__)


class NaturalProofsRetriever(BaseRetriever):
    """Retriever over the NaturalProofs dataset using FAISS.

    Encodes entries with a sentence-transformer model and stores
    vectors in a FAISS index for cosine-similarity search.

    Attributes:
        config: RetrieverConfig with dataset and path settings.
        embedding_config: EmbeddingConfig with model settings.
        ds: The loaded HuggingFace dataset.
        text_field: Name of the text field being indexed.
        model: SentenceTransformer model for encoding.
        index: FAISS index for similarity search.
    """

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        embedding_config: EmbeddingConfig | None = None,
        rebuild_index: bool = False,
    ) -> None:
        """Initialize the retriever.

        Args:
            config: Retriever configuration. Uses defaults if None.
            embedding_config: Embedding configuration. Uses defaults if None.
            rebuild_index: If True, rebuild the index even if it exists on disk.
        """
        self.config = config or RetrieverConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()

        # Load dataset
        logger.info(
            f"Loading dataset {self.config.dataset_name} ({self.config.dataset_split})"
        )
        self.ds = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            verification_mode="no_checks",
        )
        logger.info(f"Loaded {len(self.ds)} examples")
        logger.info(f"Available columns: {self.ds.column_names}")

        # Pick text field
        self.text_field = self._pick_text_field()
        logger.info(f"Using text field: {self.text_field}")

        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_config.model_name}")
        self.model = SentenceTransformer(self.embedding_config.model_name)

        # Load or build index
        if (not rebuild_index) and self._index_exists():
            logger.info("Loading existing index from disk")
            self._load_index()
        else:
            logger.info("Building new index")
            self._build_index()
            self._save_index()

    def _pick_text_field(self) -> str:
        """Select the text field to index based on available columns."""
        cols = list(self.ds.column_names)
        for c in self.config.text_field_candidates:
            if c in cols:
                return c
        raise ValueError(
            f"Could not find text field in {cols}. "
            f"Tried: {self.config.text_field_candidates}"
        )

    def _index_exists(self) -> bool:
        """Check if index and metadata files exist."""
        return (
            os.path.exists(self.config.index_path)
            and os.path.exists(self.config.meta_path)
        )

    def _iter_texts(self) -> Iterator[str]:
        """Yield normalized text for each row."""
        for row in self.ds:
            text = row[self.text_field]
            if not isinstance(text, str):
                text = str(text)
            text = text.replace("\\n", "\n")
            yield text

    def _build_index(self) -> None:
        """Embed all entries and build a FAISS index."""
        texts = list(self._iter_texts())
        logger.info(f"Encoding {len(texts)} entries...")

        self.embeddings = self.model.encode(
            texts,
            batch_size=self.embedding_config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.normalize,
        ).astype("float32")

        dim = self.embeddings.shape[1]
        logger.info(f"Embedding dimension: {dim}")

        # Use IndexFlatIP for cosine similarity (with normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        logger.info("FAISS index built and populated")

    def _save_index(self) -> None:
        """Persist FAISS index and metadata to disk."""
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(self.config.index_path), exist_ok=True)

        logger.info(f"Saving FAISS index to {self.config.index_path}")
        faiss.write_index(self.index, str(self.config.index_path))

        meta = {
            "dataset_name": self.config.dataset_name,
            "split": self.config.dataset_split,
            "text_field": self.text_field,
            "num_rows": len(self.ds),
            "embedding_dim": int(self.embeddings.shape[1]),
        }
        logger.info(f"Saving metadata to {self.config.meta_path}")
        with open(self.config.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk."""
        with open(self.config.meta_path, "r") as f:
            meta = json.load(f)

        if (
            meta["dataset_name"] != self.config.dataset_name
            or meta["split"] != self.config.dataset_split
        ):
            raise ValueError(
                "Metadata mismatch:\n"
                f"  Index built for: {meta['dataset_name']} ({meta['split']})\n"
                f"  Current: {self.config.dataset_name} ({self.config.dataset_split})"
            )

        logger.info(f"Metadata loaded: {meta}")
        self.text_field = meta["text_field"]
        self.index = faiss.read_index(str(self.config.index_path))

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return top-k entries matching the query by cosine similarity.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by relevance (highest first).
        """
        if k <= 0:
            return []

        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.embedding_config.normalize,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            row = self.ds[int(idx)]
            results.append(
                SearchResult(
                    idx=int(idx),
                    score=float(score),
                    text=row[self.text_field],
                )
            )

        return results

    def get_row(self, idx: int) -> dict:
        """Access the raw dataset row by index.

        Args:
            idx: Index of the row to retrieve.

        Returns:
            Dictionary containing all fields of the row.
        """
        return self.ds[int(idx)]
