#converts videos to MP3
import os
import subprocess

import os

files = os.listdir("Videos")

for file in files:

    if file.startswith("."):
        continue

    tutorial_number = " ".join(file.split(" ")[:2])

    print(tutorial_number,file)
    subprocess.run(["ffmpeg", "-i", f"Videos/{file}", f"audios/{tutorial_number}.mp3"])
    