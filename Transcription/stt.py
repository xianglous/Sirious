import json
from google.cloud import speech
import os


def transcribe():
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    audio = speech.RecognitionAudio(dict(uri='gs://sirious_audio/Example.flac'))

    diarization_config = speech.SpeakerDiarizationConfig(dict(
        enable_speaker_diarization=True,
        min_speaker_count=1,
        max_speaker_count=10,
    ))

    client = speech.SpeechClient()

    # audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(dict(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        language_code="en-US",
        audio_channel_count=2,
        diarization_config=diarization_config,
    ))
    print("Recognizing...")
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=100000)

    out_dict = {}

    # Get full transcription
    trans_dict = []
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        trans_dict.append({
            "transcript": result.alternatives[0].transcript,
            "confidence": result.alternatives[0].confidence,
        })

    out_dict['transcription'] = trans_dict

    # Get speaker info
    speaker_dict = []
    result = response.results[-1]
    words_info = result.alternatives[0].words
    for word_info in words_info:
        speaker_dict.append({
            "word": word_info.word,
            "speaker_tag": word_info.speaker_tag,
        })

    out_dict["speakers"] = speaker_dict

    with open("transcripts/output.json", "w") as outfile:
        json.dump(out_dict, outfile, indent=4)


def s2t(gcs_uri):
    """
    With punctuation
    """
    # os.system('export GOOGLE_APPLICATION_CREDENTIALS="./service-account-key.json"')

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(dict(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        audio_channel_count=2,
        language_code='en-US',
        enable_automatic_punctuation=True
    ))

    operation = client.long_running_recognize(config=config, audio=audio)

    print('Waiting for operation to complete...')
    response = operation.result(timeout=2 ** 14)

    print('Writing output to file...')
    output = [dict(
        transcript=res.alternatives[0].transcript,
        confidence=res.alternatives[0].confidence
    ) for res in response.results]

    fnm = 'Transcription/transcripts/output, with punc.json'
    open(fnm, 'a').close()  # Create file in OS
    with open(fnm, "w") as f:
        json.dump(dict(transcription=output), f, indent=4)


def main():
    # transcribe()
    s2t('gs://sirious_audio/Example.flac')


if __name__ == "__main__":
    main()

    # fnm = 'Transcription/transcripts/output, with punc.json'
    # open(fnm, 'a').close()  # Create file in OS
    # with open(fnm, "w") as f:
    #     json.dump(dict(transcription='hello world'), f, indent=4)
