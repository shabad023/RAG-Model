# Now you have audio, but a computer still can't search through sound. So this step listens to each MP3 and writes down what was said (transcription), then breaks it into small pieces.
import os
import json
import whisper
# json is a way of saving structured data (lists, labels, values) into a file neatly

os.makedirs("jsons", exist_ok=True)
# Creates a folder called jsons to store the results. exist_ok=True means "if the folder already exists, don't crash, just carry on."

model = whisper.load_model("small", device="mps")

audio_files = os.listdir("audios")#yaha se files lo
# Gets the list of all your MP3 files from the audios folder — these are the ones File 1 created.


for file in audio_files:
    print(f"\tTranscribing: {file}")

    tutorial_number = file.replace(".mp3", "")

# Go through each audio file. tutorial_number takes the filename and removes the .mp3 part, so "Lecture 01.mp3" becomes just "Lecture 01". You'll use this as a clean label.
    audio_path = f"audios/{file}"

    result = model.transcribe(
        audio_path,
        fp16=False
    )
# This is the big moment — Whisper listens to the audio and transcribes it. The result it gives back contains two useful things: the full text, and a list of segments.
# Here's the key idea: a segment is a small piece of speech that Whisper naturally detected — usually a sentence or a phrase — and it comes with a start time and end time. So Whisper is already doing your chunking for you. You don't have to cut the text up manually; Whisper hands you natural pieces.

    # use whisper's segments as chunks (each segment = one natural chunk)
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "source": tutorial_number,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

# For every segment, you save four things: which lecture it came from (source), when in the video it starts and ends (start, end, in seconds), and the actual words (text).
# This collection of labels around each piece is what "metadata" means — extra information about the text, not the text itself. It's useful because later you can say "this answer comes from Lecture 01 at the 2-minute mark."

    chunks_with_metadata = {  #meta data matlab ki har chunk ke sath uska source, start #time, #end time aur text bhi store karna
                        
        "chunks": chunks,
        "text": result["text"]
    }

    with open(f"jsons/{tutorial_number}.json", "w") as f:
        json.dump(chunks_with_metadata, f)

    print(f"Saved: {tutorial_number}.json")

print("\nAll lectures chunked successfully!")

# You bundle the chunks and the full text together, then save it all into a .json file (one file per lecture) inside the jsons folder. json.dump is the command that writes the data to the file.
# What this file achieved: Audio in → neat JSON files of text chunks (with timestamps and labels) out.