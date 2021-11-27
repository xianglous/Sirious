from google.cloud import dialogflow_v2 as dialogflow
from tempfile import TemporaryDirectory

class Chatbot:
    def __init__(self):
        self.project_id = "sirious-chatbot-2"
        self.session_id = "me"
        self.language_code = "en-US"
        self.audio_file_path = "./Example.flac"
        self.detect_intent_audio(self.project_id, self.session_id, self.audio_file_path, self.language_code)

def detect_intent_audio(project_id, session_id, audio_file_path, language_code):
    """Returns the result of detect intent with an audio file as input.

    Using the same `session_id` between requests allows continuation
    of the conversation."""

    session_client = dialogflow.SessionsClient()

    # Note: hard coding audio_encoding and sample_rate_hertz for simplicity.
    audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 16000

    session = session_client.session_path(project_id, session_id)
    print("Session path: {}\n".format(session))

    with open(audio_file_path, "rb") as audio_file:
        input_audio = audio_file.read()

    audio_config = dialogflow.InputAudioConfig(
        audio_encoding=audio_encoding,
        language_code=language_code,
        sample_rate_hertz=sample_rate_hertz,
    )
    query_input = dialogflow.QueryInput(audio_config=audio_config)

    request = dialogflow.DetectIntentRequest(
        session=session, query_input=query_input, input_audio=input_audio,
    )
    response = session_client.detect_intent(request=request)

    print("=" * 20)
    print("Query text: {}".format(response.query_result.query_text))
    print(
        "Detected intent: {} (confidence: {})\n".format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence,
        )
    )
    print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))
