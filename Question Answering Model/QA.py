import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


MODELS = {
    "distilbert": "distilbert-base-cased-distilled-squad",
    "bert": "bert-large-uncased-whole-word-masking-finetuned-squad", 
    "albert": "ktrapeznikov/albert-xlarge-v2-squad-v2",
}

def get_pipeline(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    # model = AutoModelForQuestionAnswering.from_pretrained(MODELS[model_name])

    return pipeline("question-answering", 
                    model=MODELS[model_name],
                    tokenizer=MODELS[model_name],
                    device=0)

def qa_bot(model_name):
    qa = get_pipeline(model_name)
    while True:
        # Open and read the article
        lecture = input('Which lecture do you want to ask about?\n>')
        if lecture == 'exit':
            break
        while not os.path.isfile(f'context/Lecture_{lecture}.txt'):
            lecture = input('Please enter a valid lecture number.\n>')
            if lecture == 'exit':
                break
        if lecture == 'exit':
            break
        with open(f'text/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
            context = f.read()

        question = input("What question do you have?\n>")
        if question == 'exit':
            break
        # Generating an answer to the question in context

        answer = qa(question=question, context=context)

        # Print the answer
        # print(f"Question: {question}")
        print(f"Answer: '{answer['answer']}' with score {answer['score']}")


def test_qa(model_name):
    qa = get_pipeline(model_name)

    with open('test/questions/testing_questions.txt', 'r') as fin:
        with open(f'test/answers/{model_name}_testing_answers.txt', 'w+') as fout:
            for line in fin:
                #print(line)
                lecture, question = line.split(',')
                # question = question.rstrip()
                fout.write(f'{lecture}: {question}')
                with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
                    context = f.read()
                answer = qa(question=question.rstrip('\n'), context=context)
                fout.write(f"Answer: '{answer['answer']}' with score {answer['score']}\n")


def main():
    for model_name in MODELS:
        test_qa(model_name)


if __name__ == '__main__':
    main()