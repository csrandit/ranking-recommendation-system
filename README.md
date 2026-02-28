# 🎬 Production-Ready Movie Recommendation Engine

A modular movie recommendation system built using collaborative filtering techniques and matrix factorization (SVD), designed with clean architecture and evaluation pipeline.

---

## 🚀 Overview

This project implements multiple recommendation approaches:

- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Matrix Factorization (SVD)

The system generates personalized Top-K movie recommendations and evaluates performance using ranking-based metrics.

---

## 🧠 Recommendation Strategies

### 1️⃣ User-Based Collaborative Filtering
Finds similar users based on rating behavior and recommends items liked by similar users.

### 2️⃣ Item-Based Collaborative Filtering
Computes similarity between items and recommends similar movies.

### 3️⃣ SVD (Matrix Factorization)
Learns latent factors representing user preferences and item characteristics.

---

## 📊 Evaluation Metrics

Models are evaluated on a held-out test set using:

- Precision@K
- Recall@K
- NDCG@K

These metrics measure ranking quality rather than simple classification accuracy.

---

## ❄ Cold Start Handling

- ✔ New user cold start handled via selected liked movies.
- ❌ New item cold start not supported (inherent limitation of collaborative filtering).

---

## 🏗 Architecture

User Input  
↓  
Feature Processing  
↓  
Similarity / SVD Model  
↓  
Ranking Module  
↓  
Top-K Recommendations  
↓  
Streamlit UI  

---

## ✨ Technical Highlights

- Modular pipeline structure
- Separate evaluation module
- Config-driven setup
- TMDB API integration for movie posters
- Caching for performance optimization
- Interactive Streamlit interface

---

## 📂 Project Structure
src/
│
├── core/
├── evaluation/
├── data_preprocessing.py
├── recommender_engine.py
└── …
---

## 🎬 Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://intelligent-movie-recommender-engine.streamlit.app/)





---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py



