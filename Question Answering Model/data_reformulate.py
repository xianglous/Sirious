import json
from pathlib import Path
from transformers import DistilBertTokenizerFast

def read_VQA(file_path):
    # load in file
    path = Path("TutorialVQAData-master/videos.json")
    with open(path, 'rb') as video_f:
        video_file = json.load(video_f)
    
    path = Path(file_path)
    with open(path, 'rb') as train_f:
        train_file = json.load(train_f)
    
    contexts = []
    questions = []
    answers = []

    for video in video_file:
        video_id = video["video_id"]
        context = " ".join(video["transcript"])
        for qa in train_file:
            if qa["video_id"] == video_id:
                contexts.append(context)
                questions.append(qa["question"])
                answer_start = qa["answer_start"]
                answer_end = qa["answer_end"]
                answer = {"text": " ".join(video["transcript"][answer_start:answer_end+1]),
                          "answer_start": answer_start,
                          "answer_end": answer_end}
                answers.append(answer)
    return contexts, questions, answers

def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, int(answers[i]['answer_start'])))
        end_positions.append(encodings.char_to_token(i, int(answers[i]['answer_end'] - 1)))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

def get_data(file_path):
    contexts, questions, answers =  read_VQA(file_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(contexts, questions, truncation=True, padding=True)
    add_token_positions(encodings, answers, tokenizer)
    return contexts, questions, answers

contexts, questions, answers = get_data("TutorialVQAData-master/train.json")
#print(contexts[0], questions[0], answers[0])
