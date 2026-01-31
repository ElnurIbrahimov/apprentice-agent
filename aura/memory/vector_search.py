"""
Vector Search for Semantic Memory Retrieval

Uses simple TF-IDF for now. Can be upgraded to embeddings later.
"""

import logging
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VectorSearch:
    """Simple semantic search using TF-IDF."""

    # Stopwords to filter out
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
        'until', 'while', 'this', 'that', 'these', 'those', 'i',
        'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'whom', 'my', 'your', 'his', 'her', 'its', 'our',
        'user', 'aura', 'chat', 'message'
    }

    def __init__(self, memory_dir: str = "aura/data/memory"):
        self.memory_dir = Path(memory_dir)
        self.documents: List[Tuple[str, str]] = []  # (source, content)
        self.index_dirty = True
        self._idf = {}
        self._doc_vectors = []

        logger.info("VectorSearch initialized")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return [w for w in words if w not in self.STOPWORDS and len(w) > 2]

    def _compute_tf(self, tokens: List[str]) -> Counter:
        """Compute term frequency."""
        return Counter(tokens)

    def index_documents(self):
        """Index all memory documents."""

        self.documents = []

        # Load daily logs
        if self.memory_dir.exists():
            for log_file in sorted(self.memory_dir.glob("*.md")):
                try:
                    content = log_file.read_text(encoding='utf-8')
                    # Split into sections
                    sections = content.split("## [")
                    for section in sections[1:]:  # Skip header
                        if section.strip():
                            self.documents.append((log_file.name, section[:1000]))
                except Exception as e:
                    logger.warning(f"Could not read {log_file}: {e}")

        # Load MEMORY.md
        memory_file = self.memory_dir.parent / "MEMORY.md"
        if memory_file.exists():
            try:
                content = memory_file.read_text(encoding='utf-8')
                for line in content.split('\n'):
                    if line.strip().startswith('- '):
                        self.documents.append(("MEMORY.md", line))
            except Exception as e:
                logger.warning(f"Could not read MEMORY.md: {e}")

        if not self.documents:
            logger.debug("No documents to index")
            self.index_dirty = False
            return

        # Compute IDF
        doc_count = len(self.documents)
        word_doc_count = Counter()

        for _, content in self.documents:
            tokens = set(self._tokenize(content))
            for token in tokens:
                word_doc_count[token] += 1

        self._idf = {
            word: math.log(doc_count / (count + 1))
            for word, count in word_doc_count.items()
        }

        # Compute document vectors
        self._doc_vectors = []
        for _, content in self.documents:
            tf = self._compute_tf(self._tokenize(content))
            vector = {word: freq * self._idf.get(word, 0) for word, freq in tf.items()}
            self._doc_vectors.append(vector)

        self.index_dirty = False
        logger.info(f"Indexed {len(self.documents)} document sections")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search for relevant documents.

        Returns list of (source, content, score) tuples.
        """

        if self.index_dirty or not self.documents:
            self.index_documents()

        if not self.documents:
            return []

        # Compute query vector
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_tf = self._compute_tf(query_tokens)
        query_vector = {word: freq * self._idf.get(word, 0) for word, freq in query_tf.items()}

        # Compute similarities
        results = []
        for i, doc_vector in enumerate(self._doc_vectors):
            # Cosine similarity
            dot_product = sum(query_vector.get(w, 0) * doc_vector.get(w, 0) for w in query_vector)
            query_norm = math.sqrt(sum(v ** 2 for v in query_vector.values())) or 1
            doc_norm = math.sqrt(sum(v ** 2 for v in doc_vector.values())) or 1
            similarity = dot_product / (query_norm * doc_norm)

            if similarity > 0:
                source, content = self.documents[i]
                results.append((source, content, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:top_k]
