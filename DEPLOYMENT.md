# 🚀 Guide de Déploiement

Tu as plusieurs options pour publier ton site.

### ❓ Peux-tu utiliser juste GitHub ?
**Non, pas tout seul.** GitHub (via GitHub Pages) ne peut héberger que des sites statiques (HTML/CSS/JS).
Comme ton projet a un **backend Python intelligent (FastAPI + Gensim)** pour calculer les maths sur les mots, il faut un serveur capable d'exécuter du Python.

**Mais !** Tu peux mettre ton code sur GitHub et le connecter à un service de déploiement gratuit. C'est la méthode "Pro".

---

## Option 1 : Hugging Face Spaces (La plus simple) 🏆
Idéal pour les démos ML.

1. Crée un compte sur [huggingface.co](https://huggingface.co/).
2. Crée un **New Space** -> Choisis **Docker**.
3. Upload tes fichiers manuellement OU connecte ton Space à ton repo GitHub.
   - Les fichiers nécessaires sont : `backend/`, `frontend/`, `requirements.txt`, `Dockerfile`.

## Option 2 : GitHub + Render (Méthode Classique) 🔗
Si tu veux avoir ton code sur GitHub et que le site se mette à jour automatiquement quand tu push.

### 1. Mettre ton code sur GitHub
1. Crée un nouveau repository sur GitHub.
2. Push tout ton code dedans.

### 2. Connecter à Render (Gratuit)
1. Crée un compte sur [render.com](https://render.com/).
2. Clique sur **New +** -> **Web Service**.
3. Connecte ton compte GitHub et sélectionne ton repository.
4. Render va détecter le `Dockerfile` automatiquement.
5. Clique sur **Create Web Service**.

Render va construire ton site et te donner une URL `https://ton-projet.onrender.com`.

> ⚠️ **Attention** : Le plan gratuit de Render met le serveur en veille après 15min d'inactivité (le premier chargement sera lent). Hugging Face Spaces est souvent plus réactif pour ce genre de démo.

---

## 📊 Comparatif des Limites (Gratuit)

| Service | RAM (Mémoire) | CPU | Mise en veille | Verdict pour ce projet |
|---------|---------------|-----|----------------|------------------------|
| **Hugging Face Spaces** | **16 GB** 🚀 | 2 vCPU | Après 48h inactif | **Parfait** (très large) |
| **Render** | 512 MB ⚠️ | 0.1 vCPU | Après 15 min inactif | **Risqué** (512MB c'est juste pour le modèle) |

**Pourquoi la RAM est importante ?**
Le modèle GloVe que l'on charge pèse environ 150-200 Mo en mémoire.
- **Render (512 Mo)** : Ça passe, mais c'est serré avec le système d'exploitation et le serveur web à côté.
- **Hugging Face (16 Go)** : Tu as de la marge pour charger des modèles 50x plus gros !
