import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import requests
import json
import glob
import base64
import os


# ===== Page setup =====
st.set_page_config(page_title="✿ Lecture Q&A ✿", page_icon="🎀", layout="wide")


# ===== Load background image as base64 =====
def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


bg_image = get_base64_image("background.jpg")


# ===== Cozy neocities styling =====
def inject_css():
    bg_css = ""
    if bg_image:
        bg_css = f"""
        .stApp {{
            background:
                linear-gradient(135deg,
                    rgba(255, 230, 250, 0.75) 0%,
                    rgba(225, 210, 255, 0.75) 50%,
                    rgba(255, 220, 245, 0.75) 100%),
                url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        """
    else:
        bg_css = """
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, #ffd6ee 0%, transparent 40%),
                radial-gradient(circle at 85% 30%, #d4c2ff 0%, transparent 45%),
                linear-gradient(135deg, #ffe3f3 0%, #e8dcff 50%, #ffe9f6 100%);
            background-attachment: fixed;
        }
        """

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&family=Patrick+Hand&family=Press+Start+2P&display=swap');

    {bg_css}

    /* all text gets the cute handwritten font */
    html, body, [class*="css"], .stMarkdown, p, span, div, label, input {{
        font-family: 'Gaegu', 'Patrick Hand', cursive !important;
        color: #6a3d8f !important;
    }}

    /* headings - lavender-pink with soft glow */
    h1, h2, h3 {{
        font-family: 'Patrick Hand', cursive !important;
        color: #c95fb8 !important;
        text-shadow: 2px 2px 0px #fff3fb, 4px 4px 0px #e8d4ff;
        letter-spacing: 1px;
    }}

    /* marquee banner - cute scrolling text */
    .cozy-marquee {{
        background: linear-gradient(90deg, #ffd6ee, #e0d2ff, #ffd6ee);
        border: 3px dashed #c98fe0;
        border-radius: 16px;
        padding: 10px;
        margin-bottom: 16px;
        font-family: 'Patrick Hand', cursive;
        font-size: 22px;
        color: #8a4dba;
        box-shadow: 0 4px 0px #d6b8ff;
    }}

    /* light, airy header banner */
    .cozy-header {{
        text-align: center;
        padding: 28px;
        background: rgba(255, 252, 255, 0.85);
        border: 3px dashed #d8a5e8;
        border-radius: 28px;
        margin-bottom: 24px;
        box-shadow: 0 6px 0px #e0c2f0;
        backdrop-filter: blur(4px);
    }}
    .cozy-title {{
        font-family: 'Patrick Hand', cursive;
        font-size: 46px;
        color: #c95fb8;
        text-shadow: 2px 2px 0px #fff3fb, 4px 4px 0px #e8d4ff;
    }}
    .cozy-sub {{
        font-family: 'Gaegu', cursive;
        font-size: 22px;
        color: #9a6bbf;
        margin-top: 6px;
    }}

    /* text input - cute cream box with lavender border */
    .stTextInput input {{
        background-color: rgba(255, 248, 253, 0.95) !important;
        border: 3px solid #d8a5e8 !important;
        border-radius: 18px !important;
        padding: 12px 18px !important;
        font-size: 20px !important;
        color: #6a3d8f !important;
        box-shadow: 0 4px 0px #e8d4ff;
    }}
    .stTextInput input:focus {{
        border-color: #c95fb8 !important;
        box-shadow: 0 4px 0px #f0c4ec;
    }}

    /* sidebar - soft frosted lavender panel */
    [data-testid="stSidebar"] {{
        background: rgba(248, 240, 255, 0.92) !important;
        border-right: 3px dashed #d8a5e8;
        backdrop-filter: blur(6px);
    }}

    /* bordered containers (chunk cards) - airy lavender cards */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 252, 255, 0.9) !important;
        border: 3px solid #d8a5e8 !important;
        border-radius: 20px !important;
        box-shadow: 0 5px 0px #e8d4ff;
        padding: 6px;
        backdrop-filter: blur(4px);
    }}

    /* success message (loaded chunks) */
    [data-testid="stSidebar"] [data-testid="stAlert"] {{
        background-color: #f0e4ff !important;
        border: 2px dashed #c98fe0 !important;
        border-radius: 16px !important;
    }}

    /* slider accent - light purple */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: #b87fd8 !important;
    }}

    /* answer text - bigger for readability */
    .stMarkdown p {{
        font-size: 20px !important;
        line-height: 1.5 !important;
    }}

    /* pixel tag for sidebar */
    .pixel-tag {{
        font-family: 'Press Start 2P', cursive;
        font-size: 10px;
        color: #8a4dba;
        background: #f0e4ff;
        padding: 6px 10px;
        border-radius: 10px;
        display: inline-block;
        border: 2px solid #d8a5e8;
    }}

    /* video player - rounded lavender border */
    video {{
        border-radius: 16px !important;
        border: 3px solid #d8a5e8 !important;
    }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ===== Marquee scrolling banner =====
st.markdown("""
<div class="cozy-marquee">
    <marquee scrollamount="6">
        ♡ ✿ welcome to lecture Q&A ✿ ask me anything about your lectures ✿ ⭐ click any timestamp to jump straight to that moment ⭐ 🎀 powered by RAG + ollama + whisper 🎀 ♡ have fun studying ♡ ✦ ⋆ ｡ ˚ ☁︎ ˚ ｡ ⋆
    </marquee>
</div>
""", unsafe_allow_html=True)


# ===== Pretty header banner =====
st.markdown("""
<div class="cozy-header">
    <div class="cozy-title">⋆˚🎀 Lecture Q&A 🎀˚⋆</div>
    <div class="cozy-sub">✿ ask me anything about your lectures ✿ click a timestamp to hop right to it ✿</div>
</div>
""", unsafe_allow_html=True)


# ===== Helper functions =====

@st.cache_resource
def load_data():
    my_dicts = joblib.load("embeddings.joblib")
    return pd.DataFrame.from_records(my_dicts)


def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text
    })
    return r.json()["embeddings"]


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def find_video_file(source):
    for ext in ["mp4", "mov", "mkv", "avi", "webm"]:
        matches = glob.glob(f"Videos/{source}*.{ext}")
        if matches:
            return matches[0]
    return None


def stream_inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:1b", "prompt": prompt, "stream": True},
        stream=True
    )
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            yield data.get("response", "")
            if data.get("done"):
                break


# ===== Main UI =====

df = load_data()

with st.sidebar:
    st.markdown('<span class="pixel-tag">★ how it works ★</span>', unsafe_allow_html=True)
    st.success(f"🌸 Loaded {len(df)} chunks")
    st.markdown(
        "♡ Your question becomes numbers (embedding)\n\n"
        "♡ The most similar lecture chunks are found\n\n"
        "♡ The AI writes an answer using those chunks\n\n"
        "♡ Click any timestamp to hop to that moment"
    )
    top_results = st.slider("🎀 chunks to retrieve", 3, 10, 5)

query = st.text_input(
    "ask a question:",
    placeholder="e.g. what is residential architecture?"
)

if query:
    with st.spinner("🔍 searching through your lectures..."):
        question_embedding = create_embedding([query])[0]
        similarities = cosine_similarity(
            np.vstack(df['embedding']),
            [question_embedding]
        ).flatten()
        max_indx = similarities.argsort()[::-1][0:top_results]
        new_df = df.loc[max_indx].copy()
        new_df["score"] = similarities[max_indx]

        chunks_list = []
        for _, row in new_df.iterrows():
            chunks_list.append({
                "source": row["source"],
                "start_time": format_time(row["start"]),
                "end_time": format_time(row["end"]),
                "text": row["text"].strip()
            })
        chunks_text = json.dumps(chunks_list, indent=2)

    prompt = f'''You are a helpful assistant that answers questions about video lectures.

Below are the most relevant chunks from the lectures, each with the source video, start time, end time, and what was said:

{chunks_text}

User's question: "{query}"

Instructions for your answer:
1. Answer using ONLY the information in the chunks above.
2. Mention the lecture name and exact start_time and end_time as given. Do NOT calculate or change any timestamps.
3. Briefly explain what is taught at that point.
4. If the question is unrelated to the chunks, reply exactly:
   "Sorry, I can only answer questions related to the video content. Please ask a question about the lectures."

Now write the answer:
'''

    st.markdown("## ⋆˚ Answer")
    st.write_stream(stream_inference(prompt))

    st.markdown("## ⋆˚ Jump to the relevant moments")
    for _, row in new_df.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**🌷 {row['source']}**")
                st.markdown(f" {format_time(row['start'])} → {format_time(row['end'])}")
                st.caption(f"♡ relevance: {row['score']:.2f}")
            with col2:
                st.caption(row["text"])

            video_path = find_video_file(row["source"])
            if video_path:
                st.video(video_path, start_time=int(row["start"]))
            else:
                st.info(f"🎀 video for '{row['source']}' not found in Videos/ folder")