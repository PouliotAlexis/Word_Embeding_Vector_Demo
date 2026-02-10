"""
FastAPI Server for Word Embedding Demo
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os

from .embeddings import get_manager

app = FastAPI(
    title="Word Embedding Demo API",
    description="Interactive word embedding visualization and exploration",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ArithmeticRequest(BaseModel):
    positive: List[str]
    negative: List[str] = []
    topn: int = 10


class GraphRequest(BaseModel):
    words: List[str]
    max_intermediates: int = 3


class AnalogyRequest(BaseModel):
    word1: str  # A is to
    word2: str  # B as
    word3: str  # C is to ?
    topn: int = 5


class ClusterRequest(BaseModel):
    words: List[str]
    n_clusters: int = 3


class JourneyRequest(BaseModel):
    start: str
    end: str
    steps: int = 10


class CloudRequest(BaseModel):
    seed_words: List[str]
    expand: int = 20


class DimensionRequest(BaseModel):
    words: List[str]
    method: str = "tsne"
    n_components: int = 2


# Endpoints
@app.get("/api/health")
async def health_check():
    """Check if the API and model are ready."""
    try:
        manager = get_manager()
        return {
            "status": "healthy",
            "model": manager.model_name,
            "vocab_size": len(manager.model.key_to_index),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/word/{word}")
async def check_word(word: str):
    """Check if a word exists in the vocabulary."""
    manager = get_manager()
    exists = manager.word_exists(word)
    return {"word": word, "exists": exists}


@app.get("/api/inspect")
async def inspect_vector(
    word: str = Query(..., description="Word to inspect"),
    compare: str = Query(None, description="Optional word to compare"),
    top_n: int = Query(15, description="Number of top dimensions to show"),
):
    """Inspect the vector components of a word."""
    manager = get_manager()

    if not manager.word_exists(word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    data = manager.inspect_vector(word, top_n, compare)

    return data


@app.post("/api/arithmetic")
async def word_arithmetic(request: ArithmeticRequest):
    """
    Perform word vector arithmetic.
    Example: king - man + woman = queen
    """
    manager = get_manager()
    results = manager.arithmetic(request.positive, request.negative, request.topn)

    return {
        "positive": request.positive,
        "negative": request.negative,
        "results": [{"word": w, "similarity": s} for w, s in results],
    }


@app.get("/api/neighbors")
async def find_neighbors(
    word: str = Query(..., description="Target word"),
    max_distance: float = Query(0.5, description="Maximum cosine distance"),
    limit: int = Query(50, description="Maximum number of results"),
):
    """Find all words within a certain distance from the target word."""
    manager = get_manager()

    if not manager.word_exists(word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    results = manager.find_neighbors(word, max_distance, limit)

    return {
        "word": word,
        "max_distance": max_distance,
        "results": [{"word": w, "distance": d} for w, d in results],
    }


@app.get("/api/synonyms")
async def find_synonyms(
    word: str = Query(..., description="Target word"),
    topn: int = Query(15, description="Number of synonyms"),
    filter_same_root: bool = Query(True, description="Filter words with same root"),
):
    """Find synonyms for a word."""
    manager = get_manager()

    if not manager.word_exists(word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    results = manager.find_synonyms(word, topn, filter_same_root)

    return {
        "word": word,
        "synonyms": [{"word": w, "similarity": s} for w, s in results],
    }


@app.post("/api/graph")
async def build_graph(request: GraphRequest):
    """Build a connection graph between the given words."""
    manager = get_manager()

    if len(request.words) < 2:
        raise HTTPException(status_code=400, detail="At least 2 words are required")

    graph = manager.build_connection_graph(request.words, request.max_intermediates)

    return {"input_words": request.words, "graph": graph}


@app.post("/api/analogy")
async def solve_analogy(request: AnalogyRequest):
    """
    Solve word analogy: word1 is to word2 as word3 is to ?
    Example: man:woman :: king:? -> queen
    """
    manager = get_manager()

    results = manager.get_analogy(
        request.word1, request.word2, request.word3, request.topn
    )

    return {
        "analogy": f"{request.word1} : {request.word2} :: {request.word3} : ?",
        "results": [{"word": w, "similarity": s} for w, s in results],
    }


@app.post("/api/cluster")
async def cluster_words(request: ClusterRequest):
    """Cluster words into semantic groups."""
    manager = get_manager()

    clusters = manager.cluster_words(request.words, request.n_clusters)

    return {
        "input_words": request.words,
        "n_clusters": request.n_clusters,
        "clusters": clusters,
    }


@app.post("/api/journey")
async def semantic_journey(request: JourneyRequest):
    """Create a semantic journey between two words."""
    manager = get_manager()

    journey = manager.semantic_journey(request.start, request.end, request.steps)

    if not journey:
        raise HTTPException(status_code=404, detail="One or both words not found")

    return {
        "start": request.start,
        "end": request.end,
        "steps": request.steps,
        "journey": journey,
    }


@app.post("/api/cloud")
async def word_cloud_data(request: CloudRequest):
    """Generate 3D word cloud data."""
    manager = get_manager()

    data = manager.get_word_cloud_data(request.seed_words, request.expand)

    return {"seed_words": request.seed_words, "points": data}


@app.post("/api/reduce")
async def reduce_dimensions(request: DimensionRequest):
    """Reduce word embeddings to 2D/3D for visualization."""
    manager = get_manager()

    data = manager.reduce_dimensions(
        request.words, method=request.method, n_components=request.n_components
    )

    return {
        "method": request.method,
        "n_components": request.n_components,
        "points": data,
    }


@app.get("/api/random-words")
async def random_words(n: int = Query(10, description="Number of random words")):
    """Get random words from the vocabulary for exploration."""
    import random

    manager = get_manager()

    vocab = list(manager.model.key_to_index.keys())
    # Filter out words with special characters
    clean_vocab = [w for w in vocab if w.isalpha() and len(w) > 2]

    sample = random.sample(clean_vocab, min(n, len(clean_vocab)))

    return {"words": sample}


@app.get("/api/semantic-search")
async def semantic_search(
    query: str = Query(..., description="Search query (one or more words)"),
    topn: int = Query(20, description="Number of results"),
):
    """Search vocabulary for semantically related words."""
    manager = get_manager()

    results = manager.semantic_search(query, topn)

    if not results:
        raise HTTPException(
            status_code=404, detail="No words from query found in vocabulary"
        )

    return {
        "query": query,
        "results": [{"word": w, "similarity": s} for w, s in results],
    }


@app.get("/api/relationships")
async def word_relationships(
    word: str = Query(..., description="Center word"),
    n_per_category: int = Query(5, description="Words per category"),
):
    """Get word relationships organized by categories."""
    manager = get_manager()

    if not manager.word_exists(word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    data = manager.get_word_relationships(word, n_per_category)

    return data


class ClusterLabeledRequest(BaseModel):
    words: List[str]
    n_clusters: int = 3


@app.post("/api/cluster-labeled")
async def cluster_with_labels(request: ClusterLabeledRequest):
    """Cluster words with automatic labels."""
    manager = get_manager()

    if len(request.words) < 2:
        raise HTTPException(status_code=400, detail="At least 2 words required")

    clusters = manager.cluster_words_with_labels(request.words, request.n_clusters)

    return {
        "input_words": request.words,
        "clusters": clusters,
    }


# Serve frontend
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

if os.path.exists(frontend_path):

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/styles.css")
    async def serve_css():
        return FileResponse(
            os.path.join(frontend_path, "styles.css"), media_type="text/css"
        )

    @app.get("/app.js")
    async def serve_app_js():
        return FileResponse(
            os.path.join(frontend_path, "app.js"), media_type="application/javascript"
        )

    @app.get("/visualizations.js")
    async def serve_viz_js():
        return FileResponse(
            os.path.join(frontend_path, "visualizations.js"),
            media_type="application/javascript",
        )
