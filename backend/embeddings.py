"""
Word Embedding Manager
Handles loading, operations and analysis on word embeddings.
"""

import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from typing import List, Tuple, Dict, Optional
import os


class EmbeddingManager:
    """Manages word embeddings and provides various operations."""

    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        """
        Initialize the embedding manager.

        Args:
            model_name: Name of the gensim model to load.
                       Options: 'glove-wiki-gigaword-100' (small, fast)
                               'word2vec-google-news-300' (large, comprehensive)
        """
        self.model_name = model_name
        self.model: Optional[KeyedVectors] = None
        self._load_model()

    def _load_model(self):
        """Load the word embedding model."""
        print(f"Loading model: {self.model_name}...")
        self.model = downloader.load(self.model_name)
        print(f"Model loaded! Vocabulary size: {len(self.model.key_to_index)}")

    def word_exists(self, word: str) -> bool:
        """Check if a word exists in the vocabulary."""
        return word.lower() in self.model.key_to_index

    def get_vector(self, word: str) -> np.ndarray:
        """Get the embedding vector for a word."""
        return self.model[word.lower()]

    def _get_dim_interpretation(self, dim: int, top_k: int = 4) -> Dict:
        """Find words that maximize/minimize a specific dimension."""
        # Get the columns for this dimension
        values = self.model.vectors[:, dim]

        # Find indices of top K largest values
        top_indices = np.argsort(values)[-top_k:][::-1]

        # Find indices of top K smallest values
        bottom_indices = np.argsort(values)[:top_k]

        return {
            "positive_words": [self.model.index_to_key[i] for i in top_indices],
            "negative_words": [self.model.index_to_key[i] for i in bottom_indices],
        }

    def inspect_vector(
        self, word: str, top_n: int = 20, compare_word: str = None
    ) -> Dict:
        """
        Inspect the vector components of a word.

        Args:
            word: The word to inspect
            top_n: Number of top positive and negative dimensions to show
            compare_word: Optional word to compare vectors with

        Returns:
            Dictionary with vector info, top dimensions, and optional comparison.
        """
        if not self.word_exists(word):
            return {}

        vector = self.get_vector(word)

        # Get top positive and negative values
        positive_indices = np.argsort(vector)[::-1][:top_n]
        negative_indices = np.argsort(vector)[:top_n]

        # Process top positive dimensions with interpretation
        top_positive = []
        for i in positive_indices:
            interp = self._get_dim_interpretation(int(i), top_k=3)
            top_positive.append(
                {"dim": int(i), "value": float(vector[i]), "interpretation": interp}
            )

        # Process top negative dimensions with interpretation
        top_negative = []
        for i in negative_indices:
            interp = self._get_dim_interpretation(int(i), top_k=3)
            top_negative.append(
                {"dim": int(i), "value": float(vector[i]), "interpretation": interp}
            )

        result = {
            "word": word.lower(),
            "dimensions": len(vector),
            "mean": float(np.mean(vector)),
            "std": float(np.std(vector)),
            "min": float(np.min(vector)),
            "max": float(np.max(vector)),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "all_values": [float(v) for v in vector],
        }

        # Comparison with another word
        if compare_word and self.word_exists(compare_word):
            compare_vec = self.get_vector(compare_word)
            similarity = float(
                self.model.similarity(word.lower(), compare_word.lower())
            )

            # Find dimensions where they differ most
            diff = vector - compare_vec
            diff_indices = np.argsort(np.abs(diff))[::-1][:top_n]

            result["comparison"] = {
                "word": compare_word.lower(),
                "similarity": similarity,
                "all_values": [float(v) for v in compare_vec],
                "biggest_differences": [
                    {
                        "dim": int(i),
                        "word1_value": float(vector[i]),
                        "word2_value": float(compare_vec[i]),
                        "diff": float(diff[i]),
                    }
                    for i in diff_indices
                ],
            }

        return result

    def arithmetic(
        self, positive: List[str], negative: List[str], topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Perform word arithmetic: sum of positive words minus sum of negative words.

        Example: king - man + woman = queen
                 positive=['king', 'woman'], negative=['man']

        Returns:
            List of (word, similarity) tuples sorted by similarity.
        """
        # Filter valid words
        pos = [w.lower() for w in positive if self.word_exists(w)]
        neg = [w.lower() for w in negative if self.word_exists(w)]

        if not pos:
            return []

        try:
            results = self.model.most_similar(positive=pos, negative=neg, topn=topn)
            return results
        except Exception as e:
            print(f"Arithmetic error: {e}")
            return []

    def find_neighbors(
        self, word: str, max_distance: float = 0.5, limit: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Find all words within a certain cosine distance from the target word.

        Args:
            word: Target word
            max_distance: Maximum cosine distance (0 = identical, 2 = opposite)
            limit: Maximum number of results

        Returns:
            List of (word, distance) tuples.
        """
        if not self.word_exists(word):
            return []

        # Get similar words (similarity = 1 - distance for normalized vectors)
        min_similarity = 1 - max_distance

        try:
            # Get more results than needed, then filter
            similar = self.model.most_similar(word.lower(), topn=limit * 2)
            results = [(w, 1 - sim) for w, sim in similar if sim >= min_similarity]
            return results[:limit]
        except Exception as e:
            print(f"Neighbors error: {e}")
            return []

    def find_synonyms(
        self, word: str, topn: int = 15, filter_same_root: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find synonyms for a word using embedding similarity.

        Args:
            word: Target word
            topn: Number of synonyms to return
            filter_same_root: If True, filter out words that share the same root

        Returns:
            List of (synonym, similarity) tuples.
        """
        if not self.word_exists(word):
            return []

        word_lower = word.lower()

        try:
            # Get more results to filter from
            similar = self.model.most_similar(word_lower, topn=topn * 3)

            results = []
            for w, sim in similar:
                # Skip if same word
                if w == word_lower:
                    continue

                # Filter words with same root (e.g., happy/happiness/happily)
                if filter_same_root:
                    # Check if one word contains the other (simple heuristic)
                    min_len = min(len(w), len(word_lower))
                    common_prefix = 0
                    for i in range(min_len):
                        if w[i] == word_lower[i]:
                            common_prefix += 1
                        else:
                            break

                    # Skip if they share more than 60% of the shorter word as prefix
                    if common_prefix > min_len * 0.6 and common_prefix >= 4:
                        continue

                results.append((w, sim))

                if len(results) >= topn:
                    break

            return results
        except Exception as e:
            print(f"Synonyms error: {e}")
            return []

    def find_path(
        self, word1: str, word2: str, max_steps: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find a semantic path between two words.

        Returns:
            List of (word, similarity_to_previous) representing the path.
        """
        if not self.word_exists(word1) or not self.word_exists(word2):
            return []

        w1, w2 = word1.lower(), word2.lower()
        path = [(w1, 1.0)]
        current = w1
        visited = {w1}
        
        # Liste étendue de mots génériques à éviter (adverbes, conjonctions, pronoms, prépositions...)
        stop_words = {
            # Conjonctions & adverbes de liaison
            "however", "although", "though", "even", "well", "rather", "instead",
            "therefore", "moreover", "furthermore", "nevertheless", "nonetheless",
            "meanwhile", "otherwise", "hence", "thus", "besides", "whereas",
            "whereby", "wherein", "thereof", "thereby", "accordingly",
            # Adverbes génériques
            "also", "already", "always", "never", "often", "still", "quite",
            "perhaps", "maybe", "really", "actually", "simply", "merely",
            "certainly", "probably", "apparently", "essentially", "particularly",
            "especially", "generally", "usually", "typically", "basically",
            "obviously", "clearly", "indeed", "anyway", "somehow", "somewhat",
            "virtually", "nearly", "almost", "entirely", "completely", "fairly",
            "largely", "mostly", "partly", "slightly", "roughly", "highly",
            # Verbes auxiliaires / modaux / génériques
            "would", "could", "should", "might", "must", "shall",
            "have", "been", "being", "become", "became", "seems", "seemed",
            "turned", "having", "doing", "done", "went", "gone", "came",
            "made", "said", "told", "asked", "gave", "took", "found",
            "knew", "thought", "wanted", "needed", "tried", "used",
            # Mots temporels génériques
            "when", "once", "then", "time", "times", "while", "until",
            "since", "during", "after", "before", "later", "earlier",
            "soon", "recently", "eventually", "finally", "initially",
            # Pronoms / déterminants
            "itself", "themselves", "himself", "herself", "myself",
            "another", "other", "others", "each", "every", "either",
            "neither", "both", "many", "much", "more", "most", "some",
            "several", "such", "same", "certain", "various", "whole",
            # Prépositions / mots de liaison courts
            "about", "above", "across", "against", "along", "among",
            "around", "between", "beyond", "through", "toward", "towards",
            "within", "without", "upon", "under",
            # Divers mots "fourre-tout"
            "thing", "things", "stuff", "fact", "case", "point",
            "kind", "sort", "type", "form", "part", "side",
            "area", "place", "number", "group", "level", "state",
        }

        for _ in range(max_steps):
            if current == w2:
                break
                
            # NOUVEAUTÉ 1 : Le Seuil de Tolérance (Early Stop)
            # Si on est déjà dans le champ lexical cible (> 55% de similarité), on arrête !
            if current != w1 and self.model.similarity(current, w2) > 0.55:
                break

            # On prend un peu plus de voisins (100 au lieu de 50) car on va en filtrer certains
            neighbors = self.model.most_similar(current, topn=100)

            best_word = None
            best_score = -1

            for word, sim in neighbors:
                # NOUVEAUTÉ 2 : On ignore les mots déjà visités, les petits mots et les mots de liaison
                if word in visited or len(word) <= 3 or word in stop_words or not word.isalpha():
                    continue

                # Score = similarity to target
                target_sim = self.model.similarity(word, w2)

                if target_sim > best_score:
                    best_score = target_sim
                    best_word = word

            if best_word is None:
                break

            visited.add(best_word)
            path.append((best_word, self.model.similarity(current, best_word)))
            current = best_word

        # Add target if not reached
        if current != w2:
            path.append((w2, self.model.similarity(current, w2)))

        return path

    def build_connection_graph(
        self, words: List[str], max_intermediates: int = 3
    ) -> Dict:
        """
        Build a graph connecting all given words through semantic paths.

        Returns:
            Dictionary with 'nodes' and 'edges' for visualization.
        """
        valid_words = [w.lower() for w in words if self.word_exists(w)]

        if len(valid_words) < 2:
            return {"nodes": [], "edges": []}

        G = nx.Graph()
        all_words = set(valid_words)

        # Add original words as nodes
        for word in valid_words:
            G.add_node(word, original=True)

        # Find paths between all pairs
        for i, w1 in enumerate(valid_words):
            for w2 in valid_words[i + 1 :]:
                path = self.find_path(w1, w2, max_steps=max_intermediates + 1)

                # Add path nodes and edges
                for j, (word, sim) in enumerate(path):
                    if word not in G:
                        G.add_node(word, original=False)
                        all_words.add(word)

                    if j > 0:
                        prev_word = path[j - 1][0]
                        weight = self.model.similarity(prev_word, word)
                        G.add_edge(prev_word, word, weight=float(weight))

        # Build result
        nodes = [
            {
                "id": word,
                "original": G.nodes[word].get("original", False),
                "group": valid_words.index(word) if word in valid_words else -1,
            }
            for word in G.nodes
        ]

        edges = [
            {"source": u, "target": v, "weight": G.edges[u, v]["weight"]}
            for u, v in G.edges
        ]

        return {"nodes": nodes, "edges": edges}

    def get_analogy(
        self, word1: str, word2: str, word3: str, topn: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy: word1 is to word2 as word3 is to ?

        Example: man:woman :: king:? -> queen
        """
        return self.arithmetic(positive=[word2, word3], negative=[word1], topn=topn)

    def cluster_words(
        self, words: List[str], n_clusters: int = 3
    ) -> Dict[str, List[str]]:
        """
        Cluster words into semantic groups.

        Returns:
            Dictionary mapping cluster_id to list of words.
        """
        valid_words = [w.lower() for w in words if self.word_exists(w)]

        if len(valid_words) < n_clusters:
            return {"0": valid_words}

        vectors = np.array([self.get_vector(w) for w in valid_words])

        kmeans = KMeans(n_clusters=min(n_clusters, len(valid_words)), random_state=42)
        labels = kmeans.fit_predict(vectors)

        clusters = {}
        for word, label in zip(valid_words, labels):
            key = str(label)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(word)

        return clusters

    def reduce_dimensions(
        self, words: List[str], method: str = "tsne", n_components: int = 2
    ) -> List[Dict]:
        """
        Reduce word embeddings to 2D/3D for visualization.

        Args:
            words: List of words to project
            method: 'tsne' or 'pca'
            n_components: 2 or 3

        Returns:
            List of {word, x, y, [z]} dictionaries.
        """
        valid_words = [w.lower() for w in words if self.word_exists(w)]

        if len(valid_words) < 2:
            return []

        vectors = np.array([self.get_vector(w) for w in valid_words])

        if method == "tsne":
            perplexity = min(30, len(valid_words) - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=max(5, perplexity),
                random_state=42,
            )
        else:
            reducer = PCA(n_components=n_components)

        reduced = reducer.fit_transform(vectors)

        result = []
        for i, word in enumerate(valid_words):
            point = {"word": word, "x": float(reduced[i, 0]), "y": float(reduced[i, 1])}
            if n_components == 3:
                point["z"] = float(reduced[i, 2])
            result.append(point)

        return result

    def semantic_journey(self, start: str, end: str, steps: int = 10) -> List[Dict]:
        """
        Create a smooth interpolation between two words in embedding space.

        Returns:
            List of intermediate points with closest words at each step.
        """
        if not self.word_exists(start) or not self.word_exists(end):
            return []

        vec_start = self.get_vector(start)
        vec_end = self.get_vector(end)

        journey = []
        for i in range(steps + 1):
            t = i / steps
            interpolated = (1 - t) * vec_start + t * vec_end

            # Find closest words to this point
            similar = self.model.similar_by_vector(interpolated, topn=3)

            journey.append(
                {
                    "step": i,
                    "t": t,
                    "closest_words": [
                        {"word": w, "similarity": float(s)} for w, s in similar
                    ],
                }
            )

        return journey

    def get_word_cloud_data(
        self, seed_words: List[str], expand: int = 20
    ) -> List[Dict]:
        """
        Generate data for a 3D word cloud visualization.

        Args:
            seed_words: Starting words
            expand: How many neighbors to include per seed word

        Returns:
            List of words with 3D coordinates.
        """
        all_words = set()

        for word in seed_words:
            if self.word_exists(word):
                all_words.add(word.lower())
                neighbors = self.model.most_similar(word.lower(), topn=expand)
                all_words.update(w for w, _ in neighbors)

        return self.reduce_dimensions(list(all_words), method="pca", n_components=3)

    def semantic_search(self, query: str, topn: int = 20) -> List[Tuple[str, float]]:
        """
        Search the vocabulary for words semantically related to a query.
        The query can be a single word or multiple words (averaged).

        Args:
            query: Query string (single word or space-separated words)
            topn: Number of results to return

        Returns:
            List of (word, similarity) tuples.
        """
        words = query.lower().split()
        valid_words = [w for w in words if self.word_exists(w)]

        if not valid_words:
            return []

        try:
            if len(valid_words) == 1:
                results = self.model.most_similar(valid_words[0], topn=topn)
            else:
                # Average the vectors of all query words
                vectors = [self.get_vector(w) for w in valid_words]
                avg_vector = np.mean(vectors, axis=0)
                results = self.model.similar_by_vector(
                    avg_vector, topn=topn + len(valid_words)
                )
                # Filter out query words from results
                results = [(w, s) for w, s in results if w not in valid_words][:topn]

            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []

    def get_word_relationships(self, word: str, n_per_category: int = 5) -> Dict:
        """
        Get different types of relationships for a word, organized by category.

        Returns:
            Dictionary with categories of related words.
        """
        if not self.word_exists(word):
            return {}

        word_lower = word.lower()

        try:
            # Get many similar words
            similar = self.model.most_similar(word_lower, topn=50)

            # Categorize by similarity strength
            very_close = []
            close = []
            related = []

            for w, sim in similar:
                if sim > 0.7:
                    very_close.append({"word": w, "similarity": float(sim)})
                elif sim > 0.5:
                    close.append({"word": w, "similarity": float(sim)})
                else:
                    related.append({"word": w, "similarity": float(sim)})

            # Also find opposites (words with low similarity to the word)
            # We'll use arithmetic: find words similar to the opposite direction
            opposite_candidates = []
            try:
                # Try to find antonyms by looking at words that are different
                all_similar = self.model.most_similar(word_lower, topn=200)
                if len(all_similar) > 100:
                    # Words at the far end of similarity might be less related
                    far_words = all_similar[100:150]
                    opposite_candidates = [
                        {"word": w, "similarity": float(sim)}
                        for w, s in far_words[:n_per_category]
                    ]
            except:
                pass

            return {
                "center": word_lower,
                "very_similar": very_close[:n_per_category],
                "similar": close[:n_per_category],
                "related": related[:n_per_category],
                "distant": opposite_candidates,
            }
        except Exception as e:
            print(f"Relationships error: {e}")
            return {}

    def cluster_words_with_labels(
        self, words: List[str], n_clusters: int = 3
    ) -> List[Dict]:
        """
        Cluster words and provide a representative label for each cluster.

        Returns:
            List of clusters with label, words, and color.
        """
        valid_words = [w.lower() for w in words if self.word_exists(w)]

        if len(valid_words) < n_clusters:
            return [{"label": "All Words", "words": valid_words, "color_index": 0}]

        vectors = np.array([self.get_vector(w) for w in valid_words])

        # Determine optimal number of clusters
        actual_clusters = min(n_clusters, len(valid_words))
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        # Group words by cluster
        cluster_groups = {}
        for word, label in zip(valid_words, labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(word)

        # Find representative word for each cluster (closest to centroid)
        result = []
        for cluster_id, cluster_words in cluster_groups.items():
            centroid = kmeans.cluster_centers_[cluster_id]

            # Find word closest to centroid
            best_word = cluster_words[0]
            best_dist = float("inf")
            for w in cluster_words:
                dist = np.linalg.norm(self.get_vector(w) - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_word = w

            result.append(
                {
                    "label": best_word.upper(),
                    "words": cluster_words,
                    "color_index": int(cluster_id),
                }
            )

        return result


# Singleton instance
_manager: Optional[EmbeddingManager] = None


def get_manager() -> EmbeddingManager:
    """Get or create the singleton EmbeddingManager."""
    global _manager
    if _manager is None:
        _manager = EmbeddingManager()
    return _manager
