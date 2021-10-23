import json
from google.cloud import speech

def transcribe():
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    speech_file = "Videos/Example.mp4"

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=1,
        max_speaker_count=10,
    )

    client = speech.SpeechClient()

    # audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        diarization_config=diarization_config,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    out_dict = []
    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        out_dict.append({})
        out_dict[-1]["Transcript"] = result.alternatives[0].transcript
        out_dict[-1]["Confidence"] = result.alternatives[0].confidence
    
    with open("output.json", "w") as outfile:
        json.dump(out_dict, outfile)

def main():
    transcribe()

if __name__ == "__main__":
    main()