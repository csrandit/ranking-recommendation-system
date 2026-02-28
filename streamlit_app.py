import streamlit as st
import requests
import os
import re
from src.core.recommender_engine import RecommenderEngine
from dotenv import load_dotenv

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Movie Recommender",
    layout="wide"
)

# =====================================================
# GLOBAL STYLING (Dark Theme)
# =====================================================

st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0E1117;
    color: white;
}

.stMultiSelect label {
    color: white !important;
}

div.stButton > button {
    background-color: #E50914;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
    height: 3.2em;
    width: 100%;
    border: none;
    transition: 0.3s ease-in-out;
}

div.stButton > button:hover {
    background-color: #b20710;
    transform: scale(1.03);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TMDB API
# =====================================================

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not found.")
    st.stop()

# =====================================================
# LOAD ENGINE
# =====================================================

@st.cache_resource
def initialize_engine():
    service = RecommenderEngine()
    service.load()
    return service

engine = initialize_engine()

# =====================================================
# FETCH POSTER
# =====================================================

@st.cache_data
def fetch_movie_poster(title: str):
    clean_title = re.sub(r"\(\d{4}\)", "", title).strip()

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": clean_title}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            for result in data["results"]:
                if result.get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"

    return "https://via.placeholder.com/300x450.png?text=No+Image"

# =====================================================
# HERO SECTION (Local Image)
# =====================================================

st.image("assets/Hero.jpg", use_container_width=True)

# =====================================================
# MOVIE SELECTION
# =====================================================

st.subheader("Select at least 3 movies you like!")

user_selection = st.multiselect(
    "",
    engine.movies_df["title"].tolist()
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# RECOMMEND BUTTON
# =====================================================

recommend_clicked = st.button("Get My Recommendations")

if recommend_clicked:

    if len(user_selection) < 3:
        st.warning("Please select at least 3 movies.")
    else:

        with st.spinner("Analyzing your taste... 🤖"):

            selected_ids = engine.movies_df[
                engine.movies_df["title"].isin(user_selection)
            ]["item_id"].tolist()

            results = engine.recommend_for_new_user(selected_ids, k=9)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Personalized Recommendations")

        cols = st.columns(3)

        for idx, (item_id, score) in enumerate(results):
            recommended_title = engine.movies_df[
                engine.movies_df["item_id"] == item_id
                ]["title"].values[0]

            poster = fetch_movie_poster(recommended_title)

            with cols[idx % 3]:
                st.image(poster, use_container_width=True)

                st.markdown(f"""
                        <div style="text-align:center">
                            <h4>{recommended_title}</h4>
                            <p>⭐ Predicted Score: {score:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)