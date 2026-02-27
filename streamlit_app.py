import streamlit as st
import requests
import os
import re

from src.data_preprocessing import MovieLensDataLoader
from src.models.user_cf import UserBasedCF


# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="AI Movie Recommender",
    layout="wide"
)

# ======================================
# DARK MODE STYLE
# ======================================

st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .stApp {
            background-color: #0E1117;
        }
    </style>
""", unsafe_allow_html=True)

# ======================================
# TOP BANNER
# ======================================

st.image(
    "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
    width="stretch"
)

st.markdown(
    "<h1 style='text-align: center;'>🎬 AI Movie Recommender</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Select at least 3 movies you like and get smart recommendations.</p>",
    unsafe_allow_html=True
)

# ======================================
# TMDB API KEY
# ======================================

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not found. Please set it as an environment variable.")
    st.stop()

# ======================================
# LOAD MODEL + DATA (Cached)
# ======================================

@st.cache_resource
def load_recommender_system():
    data_loader = MovieLensDataLoader("data/raw/ml-100k")

    ratings_dataframe = data_loader.load_ratings()
    train_dataframe, _ = data_loader.train_test_split(ratings_dataframe)

    recommender_model = UserBasedCF(k_neighbors=40)
    recommender_model.fit(train_dataframe)

    movies_dataframe = data_loader.load_movies()

    return recommender_model, movies_dataframe


recommender, movies_data = load_recommender_system()

# ======================================
# FETCH MOVIE POSTER
# ======================================

@st.cache_data
def fetch_movie_poster(movie_name: str):

    clean_name = re.sub(r"\(\d{4}\)", "", movie_name).strip()

    api_url = "https://api.themoviedb.org/3/search/movie"

    parameters = {
        "api_key": TMDB_API_KEY,
        "query": clean_name
    }

    response = requests.get(api_url, params=parameters)

    if response.status_code == 200:
        response_data = response.json()

        if response_data.get("results"):
            for result in response_data["results"]:
                if result.get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{result['poster_path']}"

    return "https://via.placeholder.com/300x450.png?text=No+Image"

# ======================================
# MOVIE SELECTION
# ======================================

user_selected_movies = st.multiselect(
    "🎥 Select movies you like:",
    movies_data["title"].tolist()
)

# ======================================
# RECOMMENDATION BUTTON
# ======================================

if st.button("🚀 Recommend"):

    if len(user_selected_movies) < 3:
        st.warning("Please select at least 3 movies.")
    else:

        with st.spinner("🤖 AI is analyzing your taste..."):

            selected_movie_ids = movies_data[
                movies_data["title"].isin(user_selected_movies)
            ]["item_id"].tolist()

            recommendations = recommender.recommend_for_new_user(
                selected_movie_ids,
                k=9,
                return_scores=True
            )

        st.subheader("🎯 Recommended For You")

        columns = st.columns(3)

        for index, (recommended_item_id, predicted_score) in enumerate(recommendations):

            recommended_movie_name = movies_data[
                movies_data["item_id"] == recommended_item_id
            ]["title"].values[0]

            poster_link = fetch_movie_poster(recommended_movie_name)

            recommendation_reason = user_selected_movies[0]

            with columns[index % 3]:
                st.image(poster_link, width="stretch")
                st.markdown(f"**{recommended_movie_name}**")
                st.caption(f"⭐ Predicted Score: {predicted_score:.2f}")
                st.caption(f"Because you liked {recommendation_reason}")