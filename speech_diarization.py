from pyannote.audio import Pipeline
import whisper
from scipy.io import wavfile
import numpy as np
import json
import torch
from os import listdir
from os.path import isfile, join

def diarization(file, auth_token):
    # Split audio into parts based on speaker
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=auth_token)
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device(0)) # Manually make use of GPU

    # 4. apply pretrained pipeline
    dia = pipeline(file)

    segments = []
    for turn, _, speaker in dia.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return segments

def load_audio(file):
    # Prepare wav file to correct format
    samplerate, audio = wavfile.read(file)
    audio = audio.astype("float32")
    audio = audio[:, 1] / 32768.0
    step = int(samplerate/16000)
    audio = np.array([audio[i] for i in range(0, len(audio), step)]) # change to samplerate (close to) 16 000
    return audio

def transcribe(audio, segments):
    # Transcribe each segment
    model = whisper.load_model("base")
    samplerate = 16000
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        wav_segment = audio[int(start*samplerate): int(end*samplerate)]
        transcription = model.transcribe(wav_segment)
        segment["text"] = transcription["text"]
    return segments

def main():
    auth_token = "Auth_token"  # pyannote auth_token

    audio_path = "audio/"
    transcription_path = "transcription/"
    # files = ["lex_fridman_clip5.wav"]
    files = [f for f in listdir(audio_path) if isfile(join(audio_path, f))]

    for file in files:
        audio_file = audio_path + file
        transcription_file = transcription_path + file[:-4] + "_diarization.json"

        segments = diarization(audio_file, auth_token)

        # with open(transcription_file, "w") as f:
        #     json.dump(segments, f, indent=2)

        # with open(transcription_file, 'r') as f:
        #     segments = json.load(f)

        audio = load_audio(audio_file)
        segments = transcribe(audio, segments)

        with open(transcription_file, "w", encoding='utf8') as f:
            json.dump(segments, f, indent=2)

        # with open(transcription_file, 'r', encoding='utf8') as f:
        #     segments = json.load(f)

        # for segment in segments:
        #     print(segment["text"])

if __name__ == "__main__": main()

