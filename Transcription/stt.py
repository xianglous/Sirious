import json
from google.cloud import speech

def transcribe():
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    audio = speech.RecognitionAudio(uri='gs://sirious_audio/Example.flac')

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=1,
        max_speaker_count=10,
    )

    client = speech.SpeechClient()

    # audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        language_code="en-US",
        audio_channel_count=2,
        diarization_config=diarization_config,
    )
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
        json.dump(out_dict, outfile)

def main():
    transcribe()

if __name__ == "__main__":
    main()