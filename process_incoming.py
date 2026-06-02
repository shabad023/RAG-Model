import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import numpy as np
import requests

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text
    })
    embeddings = r.json()["embeddings"]
    return embeddings

my_dicts = joblib.load("embeddings.joblib")
df = pd.DataFrame.from_records(my_dicts)  # ← add this line

incoming_query = input("Ask a question about the lectures: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]
print(new_df[["source", "text"]])  # removed "title" and "number" since your chunks don't have those columns