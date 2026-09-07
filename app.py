"""
AI Data Scientist — Prompt-Driven EDA & Visualization Bootcamp App
--------------------------------------------------------------
Students upload ANY csv file, then instead of writing pandas / matplotlib
code themselves, they type what they want in plain English. Gemini writes
the code, the app runs it live, and shows the student:
    1. The client-style question they asked
    2. The Python code Gemini generated (so they still learn to code!)
    3. The chart / table / number that code produced
    4. A short, plain-English explanation of the insight

Built for: D'SIAR TECH — Data Visualization Using AI bootcamp
"""

import base64
import contextlib
import io
import json
import re
import textwrap
import time
import traceback
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Scientist — EDA Bootcamp",
    page_icon="🎬",
    layout="wide",
)

sns.set_theme(style="whitegrid")

CLIENT_QUESTIONS = [
    "Show me the first 10 rows and tell me what each column means.",
    "How many rows and columns are in this dataset, and are there missing values?",
    "Show me the top 10 rows ranked by the most important numeric column, as a bar chart.",
    "Plot the distribution of the main numeric column as a histogram.",
    "Which category appears most often? Show it as a bar chart of the top 10.",
    "Is there a relationship between two numeric columns? Show a scatter plot with a trend line.",
    "Show me a correlation heatmap of all numeric columns.",
    "Find and show me any outliers in the main numeric column using a boxplot.",
]

SAFE_BUILTIN_BLOCKLIST = [
    "import os", "import sys", "subprocess", "open(", "__import__",
    "eval(", "exec(", "shutil", "socket", "requests.", "urllib",
    "os.system", "os.remove", "os.environ", "input(",
]


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "df": None,
        "df_name": None,
        "genai_client": None,
        "chat": None,
        "history": [],       # list of dicts describing each step (for UI + notebook export)
        "api_key": "",
        "model_name": "gemini-2.5-flash",
        "gemini_chat_history": [],  # kept implicitly by the SDK chat session
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def build_schema_summary(df: pd.DataFrame) -> str:
    """A compact text description of the dataframe for Gemini's context."""
    buf = io.StringIO()
    df.dtypes.to_frame("dtype").to_csv(buf)
    dtypes_csv = buf.getvalue()

    null_counts = df.isnull().sum()
    nulls_str = ", ".join(
        f"{c}: {n}" for c, n in null_counts.items() if n > 0
    ) or "none"

    sample = df.head(3).to_csv(index=False)
    # Truncate very wide/long sample text so we don't blow the prompt budget
    if len(sample) > 3000:
        sample = sample[:3000] + "\n...[truncated]..."

    schema = f"""
DATASET NAME: {st.session_state.df_name}
SHAPE: {df.shape[0]} rows x {df.shape[1]} columns

COLUMNS AND DTYPES:
{dtypes_csv}

MISSING VALUES (column: count): {nulls_str}

SAMPLE ROWS (first 3, csv format):
{sample}
""".strip()
    return schema


def build_system_instruction(df: pd.DataFrame) -> str:
    schema = build_schema_summary(df)
    return textwrap.dedent(f"""
    You are an AI data scientist assistant embedded inside a Streamlit app used
    in a college data-visualization bootcamp. Students type plain-English
    questions about a dataset (like talking to a junior data scientist), and
    you must respond with Python code that answers the question using the
    pandas dataframe that is ALREADY loaded in the execution environment as
    the variable `df`. Do NOT re-load or re-create the dataframe; it exists.

    Available in the execution environment: df (pandas DataFrame), pd, np,
    plt (matplotlib.pyplot), sns (seaborn). No other imports are allowed.

    Dataset context:
    {schema}

    RULES FOR YOUR CODE:
    - Never re-read a CSV, never use file I/O, never import anything.
    - If the answer is a chart: create it with matplotlib/seaborn, set a clear
      title and axis labels, and call plt.tight_layout(). Do NOT call plt.show().
    - If the answer is a table or number: assign the final answer to a variable
      named `result` (a DataFrame, Series, or plain value).
    - Keep code short, correct, and beginner-readable. Add short comments.
    - Work defensively: if a column looks numeric but might be stored as text
      (e.g. "2,711,075" or "2h 22m"), clean it first before analysis.
    - Only use column names that actually exist in the dataset above.

    RESPONSE FORMAT — you must reply with EXACTLY this structure, nothing else,
    no markdown code fences, no extra commentary outside these tags:

    ###EXPLANATION###
    <One to three short sentences, written for a non-technical client, in the
    voice of a data scientist presenting a finding.>
    ###CODE###
    <Pure python code only. No ``` fences.>
    """).strip()


def get_client():
    # IMPORTANT: the Client object must be kept alive in session_state.
    # If it's only a local variable, Python garbage-collects it on the next
    # Streamlit rerun, which closes its underlying httpx connection —
    # causing "Cannot send a request, as the client has been closed."
    # Re-use the same client across reruns; only recreate it if the API key
    # or model changed.
    key = (st.session_state.api_key, st.session_state.model_name)
    if (
        st.session_state.genai_client is None
        or st.session_state.get("_genai_client_key") != key
    ):
        st.session_state.genai_client = genai.Client(api_key=st.session_state.api_key)
        st.session_state["_genai_client_key"] = key
    return st.session_state.genai_client


def start_chat_session(df: pd.DataFrame):
    client = get_client()
    system_instruction = build_system_instruction(df)
    chat = client.chats.create(
        model=st.session_state.model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    st.session_state.chat = chat


def parse_gemini_response(text: str):
    """Pulls explanation + code out of the ###EXPLANATION###/###CODE### format,
    tolerating minor format slips (stray code fences, missing tags)."""
    text = text.strip()

    exp_match = re.search(r"###EXPLANATION###(.*?)###CODE###", text, re.DOTALL)
    code_match = re.search(r"###CODE###(.*)", text, re.DOTALL)

    explanation = exp_match.group(1).strip() if exp_match else ""
    code = code_match.group(1).strip() if code_match else text

    # Strip stray markdown fences if the model added them anyway
    code = re.sub(r"^```(?:python)?", "", code.strip())
    code = re.sub(r"```$", "", code.strip()).strip()

    if not explanation and not code_match:
        # Model ignored the format entirely; treat everything as code
        explanation = "(Gemini didn't return a separate explanation for this step.)"

    return explanation, code


def code_is_safe(code: str) -> bool:
    lowered = code.lower()
    return not any(bad in lowered for bad in SAFE_BUILTIN_BLOCKLIST)


def run_generated_code(code: str, df: pd.DataFrame):
    """Executes the generated code in a restricted namespace.
    Returns (success, output_kind, output_payload, stdout_text, error_text)."""
    if not code_is_safe(code):
        return False, "error", None, "", "Blocked: generated code used a disallowed operation."

    plt.close("all")
    local_env = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
    }
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {"__builtins__": __builtins__}, local_env)
    except Exception:
        return False, "error", None, stdout_capture.getvalue(), traceback.format_exc()

    stdout_text = stdout_capture.getvalue()

    # Priority 1: a matplotlib figure was produced
    if plt.get_fignums():
        fig = plt.gcf()
        return True, "figure", fig, stdout_text, ""

    # Priority 2: an explicit `result` variable
    if "result" in local_env and local_env["result"] is not None:
        return True, "result", local_env["result"], stdout_text, ""

    # Priority 3: just printed output
    if stdout_text.strip():
        return True, "stdout", stdout_text, stdout_text, ""

    return True, "none", None, stdout_text, ""


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    return buf.read()


def send_message_with_backoff(prompt: str, max_retries: int = 4, base_delay: float = 2.0):
    """Calls chat.send_message, retrying with exponential backoff on
    transient server-side failures (503 'model overloaded', 429 rate limit,
    etc). Raises the underlying exception if it never recovers, so the
    caller can show a friendly message instead of the app crashing."""
    last_error = None
    for i in range(max_retries):
        try:
            return st.session_state.chat.send_message(prompt)
        except genai_errors.ServerError as e:
            # 500/503 — Google's servers are overloaded or briefly down.
            last_error = e
        except genai_errors.ClientError as e:
            # 429 = rate limited; anything else (400, 401, 403...) is not
            # transient, so don't waste time retrying those.
            if getattr(e, "code", None) != 429:
                raise
            last_error = e
        delay = base_delay * (2 ** i)
        time.sleep(delay)
    raise last_error


def ask_gemini_with_retry(prompt: str, df: pd.DataFrame, max_attempts: int = 3):
    """Sends prompt to Gemini, executes the code, and if it errors, feeds the
    traceback back to Gemini asking it to fix it (shown to the student as a
    'the AI is debugging itself' moment).

    Two very different failure modes are handled here:
    - The GENERATED CODE is buggy -> fed back to Gemini to self-correct.
    - The API CALL ITSELF fails (Google's servers overloaded, rate limited,
      etc) -> retried with backoff; if it still fails, we return a single
      'api_error' step instead of letting the exception crash the app.
    """
    attempts = []
    current_prompt = prompt

    for attempt_num in range(1, max_attempts + 1):
        try:
            response = send_message_with_backoff(current_prompt)
        except Exception as e:
            attempts.append({
                "attempt": attempt_num,
                "explanation": "",
                "code": "",
                "success": False,
                "kind": "api_error",
                "payload": None,
                "stdout": "",
                "error": str(e),
            })
            break

        explanation, code = parse_gemini_response(response.text)
        success, kind, payload, stdout_text, error_text = run_generated_code(code, df)

        attempts.append({
            "attempt": attempt_num,
            "explanation": explanation,
            "code": code,
            "success": success,
            "kind": kind,
            "payload": payload,
            "stdout": stdout_text,
            "error": error_text,
        })

        if success:
            break

        current_prompt = (
            "That code raised an error when executed. Please fix it and "
            "return the SAME response format again.\n\n"
            f"ERROR:\n{error_text}\n\n"
            "Remember: df already exists, don't re-load it, don't import anything, "
            "and only use columns that exist in the dataset."
        )

    return attempts


def render_step_output(step, key_prefix):
    if step["kind"] == "figure":
        st.pyplot(step["payload"], clear_figure=False)
    elif step["kind"] == "result":
        payload = step["payload"]
        if isinstance(payload, (pd.DataFrame, pd.Series)):
            st.dataframe(payload)
        else:
            st.write(payload)
    elif step["kind"] == "stdout":
        st.text(step["payload"])
    elif step["kind"] == "error":
        st.error(f"Execution failed:\n\n{step['error']}")
    elif step["kind"] == "api_error":
        st.warning(
            "🌐 Gemini's servers didn't respond after several retries "
            "(this is usually the free-tier model being briefly overloaded "
            "or rate-limited, not a bug in your question). "
            "Wait a few seconds and re-type your question, or try the "
            "`gemini-2.5-flash-lite` model in the sidebar, which has looser "
            "free-tier limits.\n\n"
            f"Details: {step['error']}"
        )
    elif step["kind"] == "none":
        st.caption("(Code ran with no visible output.)")


# ----------------------------------------------------------------------
# Sidebar — setup
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Setup")

    st.session_state.api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.api_key,
        help="Get a free key at https://aistudio.google.com/apikey — "
             "each student should use their own key.",
    )

    st.session_state.model_name = st.selectbox(
        "Gemini model",
        options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
        index=0,
        help="Flash models are on Google's free tier. If a model name stops "
             "working, check https://ai.google.dev for the current lineup.",
    )

    st.divider()
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        if st.session_state.df_name != uploaded_file.name:
            try:
                df_loaded = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df_loaded = None

            if df_loaded is not None:
                st.session_state.df = df_loaded
                st.session_state.df_name = uploaded_file.name
                st.session_state.history = []
                st.session_state.chat = None
                st.success(f"Loaded {uploaded_file.name} ({df_loaded.shape[0]} rows, {df_loaded.shape[1]} cols)")

    st.divider()
    if st.button("🔄 Reset conversation", width="stretch"):
        st.session_state.history = []
        st.session_state.chat = None
        st.rerun()


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("🎬 AI Data Scientist — Prompt Your Way Through EDA")
st.caption(
    "Upload a dataset, then ask questions in plain English. Gemini writes the "
    "pandas / matplotlib code, this app runs it live, and shows you the chart, "
    "the code, and the insight — just like a junior data scientist presenting to a client."
)

if not st.session_state.api_key:
    st.info("👈 Paste a free Gemini API key in the sidebar to get started.")
    st.stop()

if st.session_state.df is None:
    st.info("👈 Upload a CSV file in the sidebar to begin your analysis.")
    st.stop()

df = st.session_state.df

# First-look panel
with st.expander("📋 First look at the data", expanded=len(st.session_state.history) == 0):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", int(df.isnull().sum().sum()))
    st.dataframe(df.head(10), width="stretch")

# Start (or resume) the Gemini chat session tied to this dataset
if st.session_state.chat is None:
    with st.spinner("Briefing Gemini on your dataset..."):
        start_chat_session(df)

# Quick client-style question buttons
st.markdown("**💬 Try a client-style question, or type your own below:**")
cols = st.columns(4)
quick_prompt = None
for i, q in enumerate(CLIENT_QUESTIONS):
    if cols[i % 4].button(q, key=f"quick_{i}", width="stretch"):
        quick_prompt = q

st.divider()

# Render history
for idx, step_group in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(step_group["prompt"])
    with st.chat_message("assistant"):
        for attempt in step_group["attempts"]:
            if attempt["attempt"] > 1:
                st.caption(f"🔁 Retry attempt {attempt['attempt']} (fixing an earlier error)")
            if attempt["explanation"]:
                st.markdown(attempt["explanation"])
            with st.expander("Show the code Gemini wrote", expanded=False):
                st.code(attempt["code"], language="python")
            render_step_output(attempt, key_prefix=f"{idx}_{attempt['attempt']}")

# Chat input
typed_prompt = st.chat_input("Ask a question about your data...")
final_prompt = quick_prompt or typed_prompt

if final_prompt:
    with st.chat_message("user"):
        st.write(final_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gemini is writing and running code..."):
            attempts = ask_gemini_with_retry(final_prompt, df)

        for attempt in attempts:
            if attempt["attempt"] > 1:
                st.caption(f"🔁 Retry attempt {attempt['attempt']} (fixing an earlier error)")
            if attempt["explanation"]:
                st.markdown(attempt["explanation"])
            with st.expander("Show the code Gemini wrote", expanded=False):
                st.code(attempt["code"], language="python")
            render_step_output(attempt, key_prefix=f"live_{attempt['attempt']}")

    st.session_state.history.append({
        "prompt": final_prompt,
        "attempts": attempts,
        "timestamp": datetime.now().isoformat(),
    })
    st.rerun()


# ----------------------------------------------------------------------
# Export as a real Jupyter notebook
# ----------------------------------------------------------------------
def build_notebook(history, df_name):
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(
        f"# AI-Assisted EDA — {df_name}\n\nGenerated by the D'SIAR TECH AI Data Scientist bootcamp app."
    )]

    for step in history:
        cells.append(nbf.v4.new_markdown_cell(f"### 🎯 Question\n{step['prompt']}"))
        last = step["attempts"][-1]
        if last["explanation"]:
            cells.append(nbf.v4.new_markdown_cell(f"**Insight:** {last['explanation']}"))

        code_cell = nbf.v4.new_code_cell(last["code"])
        outputs = []
        if last["kind"] == "figure":
            png_bytes = fig_to_png_bytes(last["payload"])
            b64 = base64.b64encode(png_bytes).decode()
            outputs.append(nbf.v4.new_output(
                output_type="display_data",
                data={"image/png": b64},
                metadata={},
            ))
        elif last["kind"] in ("result", "stdout") and last["payload"] is not None:
            text_repr = str(last["payload"])
            outputs.append(nbf.v4.new_output(
                output_type="stream", name="stdout", text=text_repr,
            ))
        code_cell["outputs"] = outputs
        cells.append(code_cell)

    nb["cells"] = cells
    return nbf.writes(nb)


if st.session_state.history:
    st.divider()
    try:
        notebook_str = build_notebook(st.session_state.history, st.session_state.df_name)
        st.download_button(
            "📥 Download this session as a Jupyter Notebook (.ipynb)",
            data=notebook_str,
            file_name=f"AI_EDA_{st.session_state.df_name.replace('.csv', '')}.ipynb",
            mime="application/x-ipynb+json",
        )
    except Exception as e:
        st.caption(f"(Notebook export unavailable: {e})")
