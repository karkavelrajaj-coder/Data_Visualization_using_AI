import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -----------------------------
# Load Dataset
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("movies_cleaned.csv")

    return df


df = load_data()


# -----------------------------
# Create TF-IDF Matrix
# -----------------------------

@st.cache_resource
def create_tfidf(data):

    tfidf = TfidfVectorizer()

    matrix = tfidf.fit_transform(
        data["genre"].fillna("")
    )

    return matrix


tfidf_matrix = create_tfidf(df)


# -----------------------------
# App Title
# -----------------------------

st.title("🎬 Movie Recommendation System")

st.write(
    "Select movies you have watched and get recommendations "
    "based on movie genres."
)


# -----------------------------
# Select Movies
# -----------------------------

movie_list = df["title"].tolist()

selected_movies = st.multiselect(
    "🎥 Select movies you have watched:",
    movie_list
)


# -----------------------------
# Number of Recommendations
# -----------------------------

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


# -----------------------------
# Recommendation Function
# -----------------------------

def recommend_movies(selected_movies, n=5):

    selected_indices = []

    for movie in selected_movies:

        index = df.index[df["title"] == movie][0]

        selected_indices.append(index)


    # Calculate similarity ONLY for selected movies

    selected_similarity = cosine_similarity(
        tfidf_matrix[selected_indices],
        tfidf_matrix
    )


    # Average similarity

    scores = selected_similarity.mean(axis=0)


    # Sort movies by similarity

    recommended_indices = scores.argsort()[::-1]


    recommendations = []


    for index in recommended_indices:

        # Don't recommend watched movies

        if df.iloc[index]["title"] in selected_movies:

            continue


        recommendations.append(index)


        if len(recommendations) == n:

            break


    return df.iloc[recommendations]


# -----------------------------
# Generate Recommendations
# -----------------------------

if st.button("🍿 Recommend Movies"):

    if len(selected_movies) == 0:

        st.warning(
            "Please select at least one movie."
        )

    else:

        recommendations = recommend_movies(
            selected_movies,
            num_recommendations
        )


        st.subheader("🎯 Recommended Movies")


        for _, movie in recommendations.iterrows():

            st.markdown(
                f"""
                ### 🎬 {movie['title']}

                **Genre:** {movie['genre']}  
                **IMDb Rating:** ⭐ {movie['imdb_rating']}  
                **Year:** {movie['year']}  
                **Director:** {movie['director_name']}
                """
            )

            st.divider()
