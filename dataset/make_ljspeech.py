import pandas as pd
from pydub import AudioSegment
import os

SAVE_DIR = "./ImranKhan-unverified"
FILES = [
    {
        "name": "ik_10",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio1 - 2m10s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/IK audio1-2m11s.mp3"
    },
    {
        "name": "ik_11",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio2 - 2m20s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/IK audio2-2m20s.mp3"
    },
    {
        "name": "ik_12",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio3 - 23m50s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/IK audio3-23m53s.mp3"
    },
    {
        "name": "ik_13",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio4 - 57m16s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/IK audio4-57m19s.mp3"
    },
    {
        "name": "ik_14",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio5 50m43s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/Ik audio5-50m43s.mp3"
    },
    {
        "name": "ik_15",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio6 - 42m38s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/IK audio6- 42m39s.mp3"
    },
    {
        "name": "ik_16",
        "tsv": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/IK audio7 26m37s.tsv",
        "audio": "/home/ihsan/hamza/urdu-tts/dataset/Whisper Youtube/audios/IK audio7-26m39s.mp3"
    },
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