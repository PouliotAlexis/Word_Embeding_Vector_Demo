---
title: Word Emedding Vector Demo
emoji: 📈
colorFrom: pink
colorTo: gray
sdk: docker
pinned: false
---

# 🧠 Word Embedding Explorer

An interactive web application to visualize and explore word embeddings (GloVe/Word2Vec). This tool allows you to perform vector arithmetic, find semantic relationships, cluster words, and visualize high-dimensional data in 2D/3D.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PouliotAlexis/Word_Embeding_Vector_Demo) [![Demo](https://img.shields.io/badge/Demo-Active-success)](https://pouliotalexis-word-embeding-vector-demo.hf.space) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) [![wakatime](https://wakatime.com/badge/github/PouliotAlexis/Word_Embeding_Vector_Demo.svg)](https://wakatime.com/badge/github/PouliotAlexis/Word_Embeding_Vector_Demo)

## ✨ Features

- **➕ Word Arithmetic**: Perform vector math like `king - man + woman = queen`.
- **📖 Synonym Finder**: Find semantically similar words.
- **🔍 Neighborhood Explorer**: Discover words within a specific semantic distance.
- **🕸️ Connection Graph**: Visualize paths and relationships between multiple words.
- **🎯 Analogy Solver**: Solve "A is to B as C is to ?" analogies.
- **🚀 Semantic Journey**: Interpolate between two words to see the path of meaning.
- **☁️ 3D Word Cloud**: Interactive 3D visualization of word clusters.
- **🗂️ Clustering**: Group words by meaning using K-Means clustering.
- **🔬 Vector Inspector**: Deep dive into the 100 dimensions of any word vector.

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://huggingface.co/spaces/PouliotAlexis/Word_Embeding_Vector_Demo
   cd Word_Embeding_Vector_Demo
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   # The first run will download the ~128MB model
   python -m uvicorn backend.server:app --reload --port 8000
   ```

5. **Open the application**
   Navigate to `http://localhost:8000` in your browser.

## 🐳 Docker Deployment

This project includes a `Dockerfile` for easy containerization.

```bash
# Build the image
docker build -t word-embedding-demo .

# Run the container (Access at http://localhost:7860)
docker run -p 7860:7860 word-embedding-demo
```

## 📚 API Documentation

The backend provides a REST API built with FastAPI.

- **Docs (Swagger UI)**: `/docs`
- **Health Check**: `/api/health`
- **Arithmetic**: `/api/arithmetic` (POST)
- **Synonyms**: `/api/synonyms` (GET)

## 🏗️ Architecture

- **Backend**: Python (FastAPI, Gensim, Scikit-learn, NumPy)
- **Frontend**: Vanilla JS, HTML5, CSS3, D3.js (Graphs), Three.js (3D Viz)
- **Model**: GloVe (Global Vectors for Word Representation) - `glove-wiki-gigaword-100`

