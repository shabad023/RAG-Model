#converts videos to MP3
import os#os module is used to interact with the operating system, such as creating directories and listing files
import subprocess#subprocess let your python code run other programs, in this case, ffmpeg to convert videos to audio

files = os.listdir("Videos")# videos di list bana lenda

for file in files:

    if file.startswith("."):#hidden files nu ignore karo
        continue

    tutorial_number = " ".join(file.split(" ")[:2])#pehle split karo naam nu by space,[0:2] pehle #2 words le linda

    print(tutorial_number,file)
    subprocess.run(["ffmpeg", "-i", f"Videos/{file}", f"audios/{tutorial_number}.mp3"])
    # This runs ffmpeg. In plain words it says: "take this video file (-i means input) and create an MP3 audio version in the audios folder." After this runs for all videos, your audios folder is full of MP3 files.