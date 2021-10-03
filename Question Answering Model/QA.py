import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


# model dictionary
MODELS = {
    "distilbert": "distilbert-base-cased-distilled-squad",
    "bert": "bert-large-uncased-whole-word-masking-finetuned-squad", 
    "albert": "ktrapeznikov/albert-xlarge-v2-squad-v2",
}


def get_pipeline(model_name):
    """Returns a QA pipeline using the model specified by model_name"""
    device = 0 if torch.cuda.is_available() else -1 # uses GPU if possible
    return pipeline("question-answering", 
                    model=MODELS[model_name],
                    tokenizer=MODELS[model_name],
                    device=device)


def qa_bot(model_name):
    """
    A simple cmd qa bot using model specified by model_name.
    Runs a loop to ask user for lecture (article) number and
    the question to answer. The bot will respond with an answer
    and a score for that answer.
    Enter 'exit' will exit out of the loop
    """

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

        answer = qa(question=question, context=context)
        print(f"Answer: '{answer['answer'].lstrip().rstrip()}' with score {answer['score']}")


def test_qa(model_name):
    """
    A testing function to test the model specified by model_name.
    The function uses questions in test/questions/testing_questions.txt.
    The lines in the test file looks like following:
        <lecture_number>,<question>
    The function will then write the answer and the score of
    each question to the output file test/answers/<model_name>_testing_answers.txt
    """

    qa = get_pipeline(model_name)
    print(f'Testing {model_name} model...')
    with open('test/questions/testing_questions.txt', 'r') as fin:
        with open(f'test/answers/{model_name}_testing_answers.txt', 'w+') as fout:
            num = 0
            for line in fin:
                print(f'Working on question {num}')
                lecture, question = line.split(',')
                fout.write(f'{lecture}: {question}')
                with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
                    context = f.read()
                answer = qa(question=question.rstrip('\n'), context=context)
                fout.write(f"Answer: '{answer['answer'].lstrip().rstrip()}' with score {answer['score']}\n")
                num += 1
    print('Finished')


def main():
    for model_name in MODELS:
        test_qa(model_name)


if __name__ == '__main__':
    main()