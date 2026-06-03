import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import requests
import json


# ----- Helper functions -----

def create_embedding(text):
    """Convert text into embedding numbers using Ollama."""
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text
    })
    return r.json()["embeddings"]


def format_time(seconds):
    """Convert seconds (e.g. 125.4) into '2m 5s' format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def inference(prompt):
    """Stream a response from the LLM word by word."""
    r = requests.post("http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    full_response = ""
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            word = data.get("response", "")
            print(word, end="", flush=True)
            full_response += word
            if data.get("done"):
                break
    print()  # newline at the end
    return full_response


# ----- Main flow -----

# 1. Load saved embeddings
print("Loading embeddings...")
my_dicts = joblib.load("embeddings.joblib")
df = pd.DataFrame.from_records(my_dicts)
print(f"Loaded {len(df)} chunks from your lectures.\n")

# 2. Get the user's question
incoming_query = input("Ask a question about the lectures: ").strip()

if not incoming_query:
    print("No question entered. Exiting.")
    exit()

# 3. Convert question to embedding
question_embedding = create_embedding([incoming_query])[0]

# 4. Find top matching chunks
similarities = cosine_similarity(
    np.vstack(df['embedding']),
    [question_embedding]
).flatten()

top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]

# 5. Build clean chunks list (with timestamps already formatted)
chunks_list = []
for _, row in new_df.iterrows():
    chunks_list.append({
        "source": row["source"],
        "start_time": format_time(row["start"]),
        "end_time": format_time(row["end"]),
        "text": row["text"].strip()
    })

chunks_text = json.dumps(chunks_list, indent=2)

# 6. Build the prompt
prompt = f'''You are a helpful assistant that answers questions about video lectures.

Below are the most relevant chunks from the lectures, each with the source video, start time, end time, and what was said:

{chunks_text}

User's question: "{incoming_query}"

Instructions for your answer:
1. Answer using ONLY the information in the chunks above.
2. Mention the lecture name (source) and the exact start_time and end_time as given. Do NOT calculate or change any timestamps.
3. Briefly explain what is taught at that point and guide the user to go to that timestamp in that lecture.
4. If multiple chunks are relevant, mention all of them.
5. If the question is unrelated to the chunks, reply exactly:
   "Sorry, I can only answer questions related to the video content. Please ask a question about the lectures."

Now write the answer:
'''

# 7. Save the prompt for debugging
with open("prompt_debug.txt", "w") as f:
    f.write(prompt)

# 8. Stream the LLM's answer
print("\nAnswer:\n")
response = inference(prompt)

# 9. Save the final answer
with open("response.txt", "w") as f:
    f.write(response)

print("\nDone. Answer also saved to response.txt")