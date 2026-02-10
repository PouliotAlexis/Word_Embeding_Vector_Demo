# 🧠 Word Embedding Explorer

Interactive web application to explore and visualize word embeddings with a modern, premium UI.

![Demo](https://img.shields.io/badge/Demo-10%20Features-blueviolet)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)

## ✨ Features

| Demo | Description |
|------|-------------|
| ➕ **Arithmetic** | `king - man + woman = queen` |
| 📖 **Synonyms** | Find semantically similar words |
| 🔍 **Neighbors** | Explore words within a distance |
| 🕸️ **Graph** | D3.js connection visualization |
| 🎯 **Analogy** | Solve A:B :: C:? problems |
| 🚀 **Journey** | Animated semantic transition |
| ☁️ **3D Cloud** | Three.js word cloud navigation |
| 🗂️ **Clustering** | Auto-group words by theme |
| 🔍 **Search** | Multi-word semantic search |
| 🎡 **Wheel** | Relationship strength visualization |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Word_embedding_demo.git
cd Word_embedding_demo

# Install dependencies
pip install -r requirements.txt

# Run the server
python -m uvicorn backend.server:app --reload --port 8001
```

Open **http://localhost:8001** in your browser.

> ⚠️ First launch downloads the GloVe model (~100MB). This happens only once.

## 📦 Requirements

- Python 3.8+
- FastAPI
- Gensim (for word embeddings)
- NumPy, Scikit-learn
- Modern browser (Chrome, Firefox, Edge)

## 🛠️ Tech Stack

**Backend:**
- FastAPI (REST API)
- Gensim (GloVe embeddings - 400,000 words)
- Scikit-learn (PCA, t-SNE, KMeans)

**Frontend:**
- Vanilla HTML/CSS/JS
- D3.js (2D visualizations)
- Three.js (3D word cloud)
- Modern CSS (glassmorphism, dark mode)

## 📁 Project Structure

```
Word_embedding_demo/
├── backend/
│   ├── embeddings.py   # EmbeddingManager class
│   └── server.py       # FastAPI endpoints
├── frontend/
│   ├── index.html      # Main UI
│   ├── styles.css      # Dark theme + glassmorphism
│   ├── app.js          # Application logic
│   └── visualizations.js  # D3.js + Three.js
└── requirements.txt
```

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/arithmetic` | Word vector math |
| GET | `/api/synonyms` | Find synonyms |
| GET | `/api/neighbors` | Nearby words |
| POST | `/api/graph` | Connection graph |
| POST | `/api/analogy` | Solve analogies |
| POST | `/api/journey` | Semantic journey |
| POST | `/api/cloud` | 3D cloud data |
| POST | `/api/cluster-labeled` | Word clustering |
| GET | `/api/semantic-search` | Multi-word search |
| GET | `/api/relationships` | Word relationships |

## 📄 License

MIT License

## 🙏 Credits

- [GloVe](https://nlp.stanford.edu/projects/glove/) - Word vectors
- [Gensim](https://radimrehurek.com/gensim/) - NLP library
- [D3.js](https://d3js.org/) - Visualizations
- [Three.js](https://threejs.org/) - 3D graphics
