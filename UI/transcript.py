# import json
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Xianglong\\Desktop\\UMich\\Courses\\EECS 498 Conversation AI\\Project Sirious\\Sirious\\UI\\cert\\key.json'
os.environ['GCLOUD_PROJECT'] = 'sirious'
from pathlib import Path
from google.cloud import speech, storage
# from icecream import ic
# from tempfile import TemporaryDirectory

BUCKET = 'sirious_audio'


def video2audio(video_path, dir):
    os.system('ffmpeg -i {} -c:a flac {}/output.flac'.format(video_path, dir))

def audio2flac(audio_path, dir):
    os.system('ffmpeg -i {} {}/output.flac'.format(audio_path, dir))

def delete(blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET)
    blob = bucket.blob(blob_name)
    blob.delete()

def upload(file_path, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    return blob.public_url


def transcribe(gcs_uri, filename):
    """
    With punctuation
    """
    # os.system('export GOOGLE_APPLICATION_CREDENTIALS="./service-account-key.json"')

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        audio_channel_count=2,
        language_code='en-US',
        enable_automatic_punctuation=True,
        )

    operation = client.long_running_recognize(config=config, audio=audio)

    print('Waiting for operation to complete...')
    response = operation.result(timeout=2 ** 14)

    print('Writing output to file...')
    output = [dict(
        transcript=res.alternatives[0].transcript,
        confidence=res.alternatives[0].confidence,
    ) for res in response.results]

    with open(Path('lectures')/filename, 'w') as f:
        f.write('\n'.join([res['transcript'].strip() for res in output]))

def main():
    # delete('Lecture_6.mp4')
    # print(upload('recordings/audio.flac', 'Lecture_6.flac'))
    transcribe('gs://sirious_audio/Lecture_6.flac', 'Lecture_6.txt')


if __name__ == "__main__":
    main()

    # fnm = 'Transcription/transcripts/output, with punc.json'
    # open(fnm, 'a').close()  # Create file in OS
    # with open(fnm, "w") as f:
    #     json.dump(dict(transcription='hello world'), f, indent=4)
