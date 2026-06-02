#turning text into numbers that computers understand

# A computer can't compare meaning using words directly. It doesn't know that "car" and "vehicle" are related. So we convert each chunk of text into a long list of numbers called an embedding. The magic of embeddings is that pieces of text with similar meaning get similar numbers. So later, when you ask a question, the computer turns your question into numbers too and finds the chunks whose numbers are closest — meaning, the closest in meaning.
import requests
import os
import json
import joblib
import pandas as pd
# requests lets your code send messages over the internet (or to a program running on your own computer).

def create_embedding(text):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text
    })

    embeddings = r.json()["embeddings"]
    return embeddings
    #embeddings = r.json()["embeddings"] aur return embeddings: Jo numbers server se aaye, unhe pakad kar wapas karta hai.
    
# This defines a reusable helper. It sends your text to Ollama — a program running on your own computer (localhost means "this computer," and 11434 is the door number it's listening on). You ask the bge-m3 model to turn the text into embeddings, and it sends back the list of numbers.


jsons=os.listdir("jsons")
my_dicts=[]
chunk_id=0
# Get the list of JSON files (made by File 2). my_dicts is an empty list where you'll collect everything. chunk_id is a counter that gives every single chunk a unique number, starting at 0.
for json_file in jsons:
    with open (f"jsons/{json_file}") as f:
        content=json.load(f)
        
    print(f"Creating embeddings for {json_file}")
    embeddings=create_embedding([c['text'] for c in content['chunks']])
    
    # Open each JSON file and read it. The part [c['text'] for c in content['chunks']] pulls out just the text from every chunk and makes a list of those texts. Then it sends that whole list to your helper, which returns one embedding (one list of numbers) for each chunk.

    
    for i,chunk in enumerate(content['chunks']):
        # print(chunk)
        chunk['chunk_id']=chunk_id
        chunk["embedding"]=embeddings[i]
        chunk_id+=1
        # chunk['embedding']=create_embedding(chunk['text'])
        my_dicts.append(chunk)
   
#    Now go through each chunk and attach two new things: its unique chunk_id, and its embedding (the matching numbers from the list). enumerate gives you both the position i and the chunk itself, so embeddings[i] grabs the right embedding for the right chunk. Then you add the finished chunk to my_dicts.
    

df=pd.DataFrame.from_records(my_dicts)
print(df)
    
# a=create_embedding("hello world")
# print(a)

joblib.dump(my_dicts, "embeddings.joblib")
print("Saved to embeddings.joblib")