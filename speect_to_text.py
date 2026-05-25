import os
import whisper

os.makedirs("transcripts", exist_ok=True)

model = whisper.load_model("small", device="mps")

audio_files = os.listdir("audios")

for file in audio_files:

    print(f"\ttranscribing: {file}")

    tutorial_number = file.replace(".mp3", "")

    audio_path = f"audios/{file}"

    result = model.transcribe(
        audio_path,
        fp16=False
    )

    with open(f"transcripts/{tutorial_number}.txt", "w") as f:
        f.write(result["text"])

    print(f"Saved: {tutorial_number}.txt")

print("\nAll lectures transcribed successfully!")