"""
FastAPI Server for Word Embedding Demo
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from .embeddings import get_manager
from .translation import get_translation_manager

app = FastAPI(
    title="Word Embedding Demo API",
    description="Interactive word embedding visualization and exploration",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    print("Pre-loading model...")
    get_manager()
    print("Model loaded successfully!")

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
    lang: str = "en"


class GraphRequest(BaseModel):
    words: List[str]
    max_intermediates: int = 3
    lang: str = "en"


class AnalogyRequest(BaseModel):
    word1: str  # A is to
    word2: str  # B as
    word3: str  # C is to ?
    topn: int = 5
    lang: str = "en"


class ClusterRequest(BaseModel):
    words: List[str]
    n_clusters: int = 3
    lang: str = "en"


class JourneyRequest(BaseModel):
    start: str
    end: str
    steps: int = 10
    lang: str = "en"


class CloudRequest(BaseModel):
    seed_words: List[str]
    expand: int = 20
    lang: str = "en"


class DimensionRequest(BaseModel):
    words: List[str]
    method: str = "tsne"
    n_components: int = 2
    lang: str = "en"


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
async def check_word(word: str, lang: str = Query("en", description="Language code")):
    """Check if a word exists in the vocabulary."""
    manager = get_manager()
    translator = get_translation_manager()
    
    en_word = translator.translate_to_en(word, lang)
    exists = manager.word_exists(en_word)
    
    return {"word": word, "en_word": en_word, "exists": exists}


@app.get("/api/inspect")
async def inspect_vector(
    word: str = Query(..., description="Word to inspect"),
    compare: str = Query(None, description="Optional word to compare"),
    top_n: int = Query(15, description="Number of top dimensions to show"),
    lang: str = Query("en", description="Language code"),
):
    """Inspect the vector components of a word."""
    manager = get_manager()
    translator = get_translation_manager()

    en_word = translator.translate_to_en(word, lang)
    en_compare = translator.translate_to_en(compare, lang) if compare else None

    if not manager.word_exists(en_word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' (en: '{en_word}') not found in vocabulary"
        )

    data = manager.inspect_vector(en_word, top_n, en_compare)
    
    # Optional: translate interpretation words back to lang
    if lang != "en" and "top_positive" in data:
        for item in data.get("top_positive", []) + data.get("top_negative", []):
            interp = item.get("interpretation", {})
            if "positive_words" in interp:
                interp["positive_words"] = translator.translate_from_en(interp["positive_words"], lang)
            if "negative_words" in interp:
                interp["negative_words"] = translator.translate_from_en(interp["negative_words"], lang)
    
    data["original_word"] = word
    data["lang"] = lang
    return data


@app.post("/api/arithmetic")
async def word_arithmetic(request: ArithmeticRequest):
    """
    Perform word vector arithmetic.
    Example: king - man + woman = queen
    """
    manager = get_manager()
    translator = get_translation_manager()
    
    en_pos = translator.translate_to_en(request.positive, request.lang) if request.positive else []
    en_neg = translator.translate_to_en(request.negative, request.lang) if request.negative else []
    
    results = manager.arithmetic(en_pos, en_neg, request.topn)
    
    result_words = [w for w, _ in results]
    translated_results = translator.translate_from_en(result_words, request.lang) if request.lang != "en" else result_words

    return {
        "positive": request.positive,
        "negative": request.negative,
        "results": [{"word": translated_results[i], "similarity": s, "en_word": w} for i, (w, s) in enumerate(results)],
    }


@app.get("/api/neighbors")
async def find_neighbors(
    word: str = Query(..., description="Target word"),
    max_distance: float = Query(0.5, description="Maximum cosine distance"),
    limit: int = Query(50, description="Maximum number of results"),
    lang: str = Query("en", description="Language code"),
):
    """Find all words within a certain distance from the target word."""
    manager = get_manager()
    translator = get_translation_manager()
    
    en_word = translator.translate_to_en(word, lang)

    if not manager.word_exists(en_word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    results = manager.find_neighbors(en_word, max_distance, limit)
    
    result_words = [w for w, _ in results]
    translated_results = translator.translate_from_en(result_words, lang) if lang != "en" else result_words

    return {
        "word": word,
        "en_word": en_word,
        "max_distance": max_distance,
        "results": [{"word": translated_results[i], "distance": d, "en_word": w} for i, (w, d) in enumerate(results)],
    }


@app.get("/api/synonyms")
async def find_synonyms(
    word: str = Query(..., description="Target word"),
    topn: int = Query(15, description="Number of synonyms"),
    filter_same_root: bool = Query(True, description="Filter words with same root"),
    lang: str = Query("en", description="Language code"),
):
    """Find synonyms for a word."""
    manager = get_manager()
    translator = get_translation_manager()
    
    en_word = translator.translate_to_en(word, lang)

    if not manager.word_exists(en_word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    # We might need more english synonyms since some might translate to the same word
    results = manager.find_synonyms(en_word, topn * 2 if lang != "en" else topn, filter_same_root)
    
    if lang != "en":
        result_words = [w for w, _ in results]
        translated_results = translator.translate_from_en(result_words, lang)
        
        # Deduplicate translated words
        seen = set([word.lower()])
        filtered_results = []
        for i, tw in enumerate(translated_results):
            if tw and tw not in seen:
                seen.add(tw)
                filtered_results.append({"word": tw, "similarity": results[i][1], "en_word": results[i][0]})
                if len(filtered_results) >= topn:
                    break
        final_results = filtered_results
    else:
        final_results = [{"word": w, "similarity": s} for w, s in results[:topn]]

    return {
        "word": word,
        "en_word": en_word,
        "synonyms": final_results,
    }


@app.post("/api/graph")
async def build_graph(request: GraphRequest):
    """Build a connection graph between the given words."""
    manager = get_manager()
    translator = get_translation_manager()

    if len(request.words) < 2:
        raise HTTPException(status_code=400, detail="At least 2 words are required")

    en_words = translator.translate_to_en(request.words, request.lang)
    graph = manager.build_connection_graph(en_words, request.max_intermediates)
    
    # Translate nodes back
    if request.lang != "en":
        all_node_words = [node["id"] for node in graph["nodes"]]
        translated_nodes = translator.translate_from_en(all_node_words, request.lang)
        translation_map = dict(zip(all_node_words, translated_nodes))
        
        for i, node in enumerate(graph["nodes"]):
            node["en_id"] = node["id"]
            node["id"] = translation_map.get(node["id"], node["id"])
            
        for edge in graph["edges"]:
            edge["source"] = translation_map.get(edge["source"], edge["source"])
            edge["target"] = translation_map.get(edge["target"], edge["target"])

    return {"input_words": request.words, "graph": graph}


@app.post("/api/analogy")
async def solve_analogy(request: AnalogyRequest):
    """
    Solve word analogy: word1 is to word2 as word3 is to ?
    Example: man:woman :: king:? -> queen
    """
    manager = get_manager()
    translator = get_translation_manager()
    
    en_word1 = translator.translate_to_en(request.word1, request.lang)
    en_word2 = translator.translate_to_en(request.word2, request.lang)
    en_word3 = translator.translate_to_en(request.word3, request.lang)

    results = manager.get_analogy(
        en_word1, en_word2, en_word3, request.topn
    )
    
    result_words = [w for w, _ in results]
    translated_results = translator.translate_from_en(result_words, request.lang) if request.lang != "en" else result_words

    return {
        "analogy": f"{request.word1} : {request.word2} :: {request.word3} : ?",
        "results": [{"word": translated_results[i], "similarity": s, "en_word": w} for i, (w, s) in enumerate(results)],
    }


@app.post("/api/cluster")
async def cluster_words(request: ClusterRequest):
    """Cluster words into semantic groups."""
    manager = get_manager()
    translator = get_translation_manager()
    
    en_words = translator.translate_to_en(request.words, request.lang)

    en_clusters = manager.cluster_words(en_words, request.n_clusters)
    
    if request.lang != "en":
        clusters = {}
        for k, words in en_clusters.items():
            translated = translator.translate_from_en(words, request.lang)
            clusters[k] = translated
    else:
        clusters = en_clusters

    return {
        "input_words": request.words,
        "n_clusters": request.n_clusters,
        "clusters": clusters,
    }


@app.post("/api/journey")
async def semantic_journey(request: JourneyRequest):
    """Create a semantic journey between two words."""
    manager = get_manager()
    translator = get_translation_manager()

    en_start = translator.translate_to_en(request.start, request.lang)
    en_end = translator.translate_to_en(request.end, request.lang)

    journey = manager.semantic_journey(en_start, en_end, request.steps)

    if not journey:
        raise HTTPException(status_code=404, detail="One or both words not found")
        
    if request.lang != "en":
        for step in journey:
            words = [item["word"] for item in step.get("closest_words", [])]
            translated = translator.translate_from_en(words, request.lang)
            for i, item in enumerate(step.get("closest_words", [])):
                item["en_word"] = item["word"]
                if len(translated) > i:
                    item["word"] = translated[i]

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
    translator = get_translation_manager()

    en_seeds = translator.translate_to_en(request.seed_words, request.lang)
    data = manager.get_word_cloud_data(en_seeds, request.expand)
    
    if request.lang != "en":
        words = [p["word"] for p in data]
        translated = translator.translate_from_en(words, request.lang)
        for i, p in enumerate(data):
            p["en_word"] = p["word"]
            if len(translated) > i:
                p["word"] = translated[i]

    return {"seed_words": request.seed_words, "points": data}


@app.post("/api/reduce")
async def reduce_dimensions(request: DimensionRequest):
    """Reduce word embeddings to 2D/3D for visualization."""
    manager = get_manager()
    translator = get_translation_manager()

    en_words = translator.translate_to_en(request.words, request.lang)
    data = manager.reduce_dimensions(
        en_words, method=request.method, n_components=request.n_components
    )
    
    if request.lang != "en":
        # use original input words since dimension request aligns with input
        for i, p in enumerate(data):
            p["en_word"] = p["word"]
            if len(request.words) > i:
                # Need to lookup or we just rely on order if it's preserved
                # Actually, reduce_dimensions filters missing words. 
                # Let's just translate what comes back to be safe.
                p["word"] = translator.translate_from_en(p["word"], request.lang)

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
    lang: str = Query("en", description="Language code"),
):
    """Search vocabulary for semantically related words."""
    manager = get_manager()
    translator = get_translation_manager()

    en_query = translator.translate_to_en(query, lang)
    results = manager.semantic_search(en_query, topn)

    if not results:
        raise HTTPException(
            status_code=404, detail="No words from query found in vocabulary"
        )
        
    result_words = [w for w, _ in results]
    translated_results = translator.translate_from_en(result_words, lang) if lang != "en" else result_words

    return {
        "query": query,
        "en_query": en_query,
        "results": [{"word": translated_results[i], "similarity": s, "en_word": w} for i, (w, s) in enumerate(results)],
    }


@app.get("/api/relationships")
async def word_relationships(
    word: str = Query(..., description="Center word"),
    n_per_category: int = Query(5, description="Words per category"),
    lang: str = Query("en", description="Language code"),
):
    """Get word relationships organized by categories."""
    manager = get_manager()
    translator = get_translation_manager()

    en_word = translator.translate_to_en(word, lang)

    if not manager.word_exists(en_word):
        raise HTTPException(
            status_code=404, detail=f"Word '{word}' not found in vocabulary"
        )

    data = manager.get_word_relationships(en_word, n_per_category)
    data["center_en"] = data["center"]
    data["center"] = word

    if lang != "en":
        for cat in ["very_similar", "similar", "related", "distant"]:
            if cat in data and data[cat]:
                words = [item["word"] for item in data[cat]]
                translated = translator.translate_from_en(words, lang)
                for i, item in enumerate(data[cat]):
                    item["en_word"] = item["word"]
                    if len(translated) > i:
                        item["word"] = translated[i]

    return data


class ClusterLabeledRequest(BaseModel):
    words: List[str]
    n_clusters: int = 3
    lang: str = "en"


@app.post("/api/cluster-labeled")
async def cluster_with_labels(request: ClusterLabeledRequest):
    """Cluster words with automatic labels."""
    manager = get_manager()
    translator = get_translation_manager()

    if len(request.words) < 2:
        raise HTTPException(status_code=400, detail="At least 2 words required")

    en_words = translator.translate_to_en(request.words, request.lang)
    clusters = manager.cluster_words_with_labels(en_words, request.n_clusters)
    
    if request.lang != "en":
        for cluster in clusters:
            # translate label
            cluster["en_label"] = cluster["label"]
            translated_label = translator.translate_from_en(cluster["label"], request.lang)
            cluster["label"] = translated_label.upper() if translated_label else cluster["label"]
            
            # translate words
            translated_words = translator.translate_from_en(cluster["words"], request.lang)
            cluster["en_words"] = cluster["words"]
            cluster["words"] = translated_words

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
