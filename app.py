"""
Movie Recommender — pick the movies you've watched, get genre-based
recommendations back. No API key, no LLM calls, no rate limits — pure
pandas + scikit-learn, so it works reliably for every student at once.

Technique: TF-IDF over each movie's genre tags + cosine similarity.
This weights RARE genres (Western, Horror, Film-Noir) more heavily than
common ones (Drama appears in 70% of this dataset), so recommendations
aren't just "everything is a Drama."
"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_data(path_or_buffer):
    df = pd.read_csv(path_or_buffer)

    # Be forgiving about column-name variants (imbd_rating vs imdb_rating etc.)
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if "imbd_rating" in low or ("imdb" in low and "rating" in low):
            rename_map[col] = "imdb_rating"
        elif "imbd_votes" in low or ("imdb" in low and "vote" in low):
            rename_map[col] = "imdb_votes"
    df = df.rename(columns=rename_map)

    required = ["title", "genre"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}")

    df["genre"] = df["genre"].fillna("")
    # Drop rows with no title or no genre info at all
    df = df[(df["title"].notna()) & (df["genre"].str.strip() != "")].reset_index(drop=True)

    # De-duplicate on title, just in case
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    return df


@st.cache_resource
def build_similarity_model(genre_series: pd.Series):
    """TF-IDF over genre tokens (comma-separated, e.g. 'Action,Crime,Drama')
    so rare genres carry more weight than common ones, then cosine
    similarity between every pair of movies."""
    vectorizer = TfidfVectorizer(
        tokenizer=lambda s: [g.strip() for g in s.split(",") if g.strip()],
        preprocessor=lambda s: s,
        token_pattern=None,
        lowercase=False,
    )
    tfidf_matrix = vectorizer.fit_transform(genre_series)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix


def recommend(df, similarity_matrix, watched_indices, top_n=10):
    """Average the similarity rows of every watched movie to build a
    'taste profile', then rank all NOT-watched movies by that."""
    if not watched_indices:
        return df.sort_values("imdb_rating", ascending=False).head(top_n).assign(match_score=None)

    profile_scores = similarity_matrix[watched_indices].mean(axis=0)

    candidates = df.copy()
    candidates["match_score"] = profile_scores
    candidates = candidates.drop(index=watched_indices, errors="ignore")

    # Never recommend a movie that shares literally zero genre overlap with
    # the watched movies (e.g. don't suggest a pure Drama to someone who
    # only watched Westerns just to pad the list out to top_n).
    candidates = candidates[candidates["match_score"] > 0]

    if "imdb_rating" in candidates.columns:
        candidates = candidates.sort_values(
            ["match_score", "imdb_rating"], ascending=[False, False]
        )
    else:
        candidates = candidates.sort_values("match_score", ascending=False)

    return candidates.head(top_n)


def matched_genres(row_genre: str, watched_genres: set) -> list:
    row_set = {g.strip() for g in row_genre.split(",") if g.strip()}
    return sorted(row_set & watched_genres)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.caption("Genre-based recommendations — TF-IDF + cosine similarity, no LLM needed.")

    uploaded = st.file_uploader("Use a different CSV (optional)", type=["csv"])
    st.caption(
        "CSV needs a `title` column and a `genre` column "
        "(comma-separated tags, e.g. `Action,Crime,Drama`)."
    )

    top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10)


# ----------------------------------------------------------------------
# Load & prep data
# ----------------------------------------------------------------------
try:
    df = load_data(uploaded if uploaded is not None else "movies_slim.csv")
except Exception as e:
    st.error(f"Couldn't load the dataset: {e}")
    st.stop()

vectorizer, tfidf_matrix, similarity_matrix = build_similarity_model(df["genre"])

# ----------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------
st.title("🎬 Find your next movie")
st.write("Select the movies you've already watched, and get recommendations based on genre.")

title_options = df["title"].tolist()
if "year" in df.columns:
    label_map = {row.title: f"{row.title} ({int(row.year)})" if pd.notna(row.year) else row.title
                 for row in df.itertuples()}
else:
    label_map = {t: t for t in title_options}

watched_titles = st.multiselect(
    "Movies you've watched:",
    options=title_options,
    format_func=lambda t: label_map.get(t, t),
    placeholder="Start typing a movie title...",
)

watched_indices = df.index[df["title"].isin(watched_titles)].tolist()
watched_genre_set = set()
for g in df.loc[watched_indices, "genre"]:
    watched_genre_set.update(x.strip() for x in g.split(",") if x.strip())

st.divider()

if not watched_titles:
    st.info("👆 Pick a few movies you've watched to get personalized recommendations. "
             "Showing the top-rated movies overall for now.")
else:
    st.markdown(f"**Your taste profile, based on {len(watched_titles)} movie(s):** "
                + ", ".join(f"`{g}`" for g in sorted(watched_genre_set)))

recommendations = recommend(df, similarity_matrix, watched_indices, top_n=top_n)

st.subheader("Recommended for you" if watched_titles else "Top rated movies")

if watched_titles and 0 < len(recommendations) < top_n:
    st.caption(
        f"Only {len(recommendations)} movie(s) in this dataset share a genre "
        f"with what you've watched — showing all of them."
    )
elif watched_titles and len(recommendations) == 0:
    st.warning("No other movies in this dataset share a genre with your picks.")

cols_per_row = 2
rows = [recommendations.iloc[i:i + cols_per_row] for i in range(0, len(recommendations), cols_per_row)]

for row_chunk in rows:
    cols = st.columns(cols_per_row)
    for col, (_, movie) in zip(cols, row_chunk.iterrows()):
        with col:
            with st.container(border=True):
                year_str = f" ({int(movie['year'])})" if "year" in movie and pd.notna(movie["year"]) else ""
                st.markdown(f"**{movie['title']}{year_str}**")

                meta_bits = []
                if "imdb_rating" in movie and pd.notna(movie["imdb_rating"]):
                    meta_bits.append(f"⭐ {movie['imdb_rating']}/10")
                if "duration" in movie and pd.notna(movie.get("duration")):
                    meta_bits.append(str(movie["duration"]))
                if "certificate" in movie and pd.notna(movie.get("certificate")):
                    meta_bits.append(str(movie["certificate"]))
                if meta_bits:
                    st.caption(" · ".join(meta_bits))

                st.write(movie["genre"])

                if watched_titles:
                    overlap = matched_genres(movie["genre"], watched_genre_set)
                    if overlap:
                        st.caption(f"Matched on: {', '.join(overlap)}")

                if "director_name" in movie and pd.notna(movie.get("director_name")):
                    st.caption(f"Dir. {movie['director_name']}")
