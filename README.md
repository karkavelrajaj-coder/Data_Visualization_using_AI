# 🎬 Movie Recommender

A simple, LLM-free movie recommender: pick the movies you've watched, get
genre-based recommendations back. No API keys, no rate limits, no external
calls — pure `pandas` + `scikit-learn`, so it works reliably even with a
whole classroom hitting it at once.

## How it works

Each movie's genre tags (e.g. `Action,Crime,Drama`) are turned into a
**TF-IDF vector**, which — unlike a naive "count shared genres" approach —
automatically weights *rare* genres (Western, Horror, Film-Noir) more
heavily than common ones. In this dataset 71% of movies are tagged Drama,
so a naive approach would recommend Drama for almost anyone; TF-IDF avoids
that.

Once a student picks their watched movies, their "taste profile" is the
average of those movies' similarity rows against every other movie
(cosine similarity). Recommendations are ranked by that similarity score,
tie-broken by IMDb rating, and movies with **zero genre overlap** are
filtered out entirely rather than padding the list.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

```bash
git init
git add .
git commit -m "Movie recommender"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Then on https://share.streamlit.io → "New app" → pick the repo → main file
`app.py` → Deploy.

## Dataset

`movies_slim.csv` is a trimmed copy (title, year, genre, rating, runtime,
certificate, director) of the original `movies.csv` — the full file
includes review text and isn't needed for genre-based recommendations, so
it's left out to keep the repo small and the app fast. Students can also
upload a different CSV from the sidebar as long as it has `title` and
`genre` columns (comma-separated genre tags).
