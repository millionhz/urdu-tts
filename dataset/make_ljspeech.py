import pandas as pd
from pydub import AudioSegment
import os

SAVE_DIR = "/content/MaryamNawazDataset"
FILES = [
    {
        "name": "mns10",
        "tsv": "/content/Whisper Youtube/MN audio1 1hour.tsv",
        "audio": "/content/Whisper Youtube/audios/MN audio2-1hr19s.mp3"
    },
    {
        "name": "mns11",
        "tsv": "/content/Whisper Youtube/MN audio2 11m 44s.tsv",
        "audio": "/content/Whisper Youtube/audios/MN audio1-11m55s.mp3"
    }
]

def make_ljspeech(save_dir: str, files):
    metadata = []

    metadata_dir = os.path.join(save_dir, "metadata.csv")
    wavs_dir = os.path.join(save_dir, "wavs")

    os.makedirs(save_dir)
    os.makedirs(wavs_dir)

    for data in files:
        name = data["name"]
        tsv_dir = data["tsv"]
        audio_dir = data["audio"]

        df = pd.read_csv(tsv_dir, sep="\t")
        audio = AudioSegment.from_mp3(audio_dir)

        for index, row in df.iterrows():
            start_time = int(row['start'])
            end_time = int(row['end'])
            audio_segment = audio[start_time:end_time]

            filename = f'{name}_{index}.wav'
            save_path = os.path.join(wavs_dir, filename)
            audio_segment.export(save_path, format='wav')

            metadata_entry = f'{filename}|{row["text"]}|'
            metadata.append(metadata_entry)

    with open(metadata_dir, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(entry + '\n')

make_ljspeech(SAVE_DIR, FILES)