# 🎬 AI Data Scientist — EDA Bootcamp App

A Streamlit app where students upload a CSV and do EDA + visualization by
**prompting Gemini in plain English** instead of writing pandas/matplotlib
code directly. Every answer shows the student:

1. The question they asked (client-style, like the Colab notebook bootcamp)
2. The Python code Gemini wrote to answer it
3. The chart / table / number that code produced
4. A short plain-English insight

Students can also **download the whole session as a real `.ipynb` notebook**
at the end — a nice artifact to submit or keep.

---

## 1. Get a free Gemini API key

Each student should get their **own** free key (keeps things free and avoids
one shared key hitting rate limits for the whole class):

1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account → "Create API key"
3. Paste it into the app's sidebar (it is never stored on a server — it only
   lives in that browser session)

## 2. Run it locally first (optional but recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, upload `movies.csv` (or any CSV), paste
your API key in the sidebar, and try a few prompts.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "AI EDA bootcamp app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## 4. Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub
2. Click **"New app"**
3. Pick your repo, branch `main`, and main file path `app.py`
4. Click **Deploy**

That's it — Streamlit Cloud installs `requirements.txt` automatically. Share
the resulting `https://<something>.streamlit.app` link with your students.
No server-side API key is needed since each student pastes their own.

---

## How the app works (for your own reference)

- On upload, the app builds a short **schema summary** of the CSV (columns,
  dtypes, null counts, 3 sample rows) and gives that to Gemini as a
  `system_instruction` inside a `google-genai` chat session — so Gemini
  always knows the columns really available and keeps memory of earlier
  turns in the same session (so "now filter to just Drama movies" works as
  a follow-up).
- Gemini is instructed to reply in a strict
  `###EXPLANATION### ... ###CODE### ...` format so the app can separate the
  human-readable insight from the runnable code.
- The code is executed in a namespace that only exposes `df`, `pd`, `np`,
  `plt`, `sns` — no file I/O, no `import os`, no `eval`/`exec`, blocked by a
  simple keyword filter. This is fine for a classroom tool where students
  bring their own (non-sensitive) datasets; it is **not** a hardened sandbox
  for untrusted production use.
- If the generated code throws an error, the traceback is sent back to
  Gemini automatically (up to 3 attempts) asking it to fix its own code —
  students get to see the AI "debug itself" live, which is a good teaching
  moment.
- At the end of the session, `nbformat` reassembles every question, the
  final working code, and the resulting chart/table into a real
  downloadable `.ipynb` file.

## Notes on the Gemini model name

Google renames / retires Flash model versions every few months. The sidebar
model dropdown defaults to `gemini-2.5-flash`. If a model stops working,
check https://ai.google.dev for the current free-tier model names and
either pick another one from the dropdown or edit the list in `app.py`.

## Customizing the quick-prompt buttons

`CLIENT_QUESTIONS` near the top of `app.py` are generic on purpose so the
app works with *any* uploaded CSV. If you want to hard-code the exact 15
"client questions" from your movies-dataset Colab notebook for a guided
walkthrough, just replace that list with your own prompts.
