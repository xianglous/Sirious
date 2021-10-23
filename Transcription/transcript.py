import json

def get_transcript():
    transcript = []
    with open('transcripts/output.json') as json_file:
        data = json.load(json_file)
    for line in data["transcription"]:
        transcript.append(line['transcript'])
    with open('transcripts/example.txt', 'w') as f:
        f.write('\n'.join(transcript))
    
def get_speaker_transcript():
    # transcripts = {}
    with open('transcripts/output.json') as json_file:
        data = json.load(json_file)
    last_speaker = -1
    with open('transcripts/speakers.txt', 'w') as f:
        for word_info in data["speakers"]:
            speaker = word_info['speaker_tag']
            if last_speaker == -1 or last_speaker != speaker:
                f.write(f"\nSpeaker {speaker}: {word_info['word']}")
            else:
                f.write(f" {word_info['word']}")
            last_speaker = speaker
    
def main():
    get_speaker_transcript()

if __name__ == '__main__':
    main()