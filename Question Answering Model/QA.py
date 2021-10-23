import os
import torch
# import spacy
# import neuralcoref
from string import punctuation
# from stanza.server import CoreNLPClient
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


# nlp = spacy.load('en')
# neuralcoref.add_to_pipe(nlp)

# model dictionary
MODELS = {
    "xlnet": "jkgrad/xlnet-base-cased-squad-quoref",
    "electra": "ahotrod/electra_large_discriminator_squad2_512",
    "roberta": "navteca/roberta-base-squad2",
    "bert": "bert-large-uncased-whole-word-masking-finetuned-squad", 
    "distilbert": "distilbert-base-cased-distilled-squad",
    "albert": "ktrapeznikov/albert-xlarge-v2-squad-v2",
    "bart": "yjernite/bart_eli5",
    "t5-abs": "tuner007/t5_abs_qa",
    "t5-qa-qg": "valhalla/t5-small-qa-qg-hl"
}


def get_t2t_pipeline(model_name):
    device = 0 if torch.cuda.is_available() else -1 # uses GPU if possible
    return pipeline("text2text-generation", 
                    model=MODELS[model_name],
                    tokenizer=AutoTokenizer.from_pretrained(MODELS[model_name]),
                    device=device)
    # text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")


def get_t2t_answer(t2t, question, context):
    answer = t2t(f"question: {question} context: {context} </s>")
    return answer


def t2t_bot():

    t2t = get_t2t_pipeline('t5-qa-qg')

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
        with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
            context = f.read()

        question = input("What question do you have?\n>")
        if question == 'exit':
            break

        answer = get_t2t_answer(t2t, question, context.split('\n\n\n')[0])
        print(answer)
        # print(f"Answer: '{answer['answer'].lstrip().rstrip()}' with score {answer['score']}")    

"""
def count_punct(seq):
    count = 0
    for token in seq:   
        for c in token.word:
            if c in punctuation:  
                count = count + 1
    return count


def moreRepresentativeThan(m1, m2, c1, c2):
    Adapted from the Java code
    if (m1.endIndex - m1.beginIndex > 10):
        return False
    if (m2 is None) or (m2.endIndex - m2.beginIndex > 10):
        return True
    if m1.mentionType != m2.mentionType:
        if (m1.mentionType == 'PROPER' and m2.mentionType != 'PROPER')\
            or (m1.mentionType == 'NOMINAL' and m2.mentionType == 'PRONOMINAL'):
            return True
        else:
            return False
    else:
        if c1 > c2:
            return False
        # First, check length
        if (m1.headIndex - m1.beginIndex > m2.headIndex - m2.beginIndex):
            return True
        if (m1.headIndex - m1.beginIndex < m2.headIndex - m2.beginIndex):
            return False
        if (m1.endIndex - m1.beginIndex > m2.endIndex - m2.beginIndex):
            return True
        if (m1.endIndex - m1.beginIndex < m2.endIndex - m2.beginIndex):
            return False
        # Now check relative position
        if (m1.sentenceIndex < m2.sentenceIndex): return True
        if (m1.sentenceIndex > m2.sentenceIndex): return False
        if (m1.headIndex < m2.headIndex): return True
        if (m1.headIndex > m2.headIndex): return False
        if (m1.beginIndex < m2.beginIndex): return True
        if (m1.beginIndex > m2.beginIndex): return False
        # At this point they're equal...
        return False
"""

def get_pipeline(model_name):
    """Returns a QA pipeline using the model specified by model_name"""
    device = 0 if torch.cuda.is_available() else -1 # uses GPU if possible
    # model = AutoModelForQuestionAnswering.from_pretrained(MODELS[model_name])
    # tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    return pipeline("question-answering", 
                    model=MODELS[model_name],
                    tokenizer=MODELS[model_name],
                    device=0)


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
        while not os.path.isfile(f'context/Lecture_{lecture}_allen_resolved.txt'):
            lecture = input('Please enter a valid lecture number.\n>')
            if lecture == 'exit':
                break
        if lecture == 'exit':
            break
        with open(f'context/Lecture_{lecture}_allen_resolved.txt', 'r', encoding="utf-8") as f:
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
    with open('test/questions/testing_questions.txt', 'r', encoding="utf-8") as fin:
        with open(f'test/answers/{model_name}_testing_answers.txt', 'w+', encoding="utf-8") as fout:
            num = 0
            for line in fin:
                print(f'Working on question {num}')
                lecture, question = line.split(',')
                fout.write(f'{lecture}: {question}')
                with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
                    context1 = f.read()
                with open(f'context/Lecture_{lecture}_resolved.txt', 'r', encoding="utf-8") as f:
                    context2 = f.read()
                with open(f'context/Lecture_{lecture}_nc_resolved.txt', 'r', encoding="utf-8") as f:
                    context3 = f.read()
                with open(f'context/Lecture_{lecture}_clustering_resolved.txt', 'r', encoding="utf-8") as f:
                    context4 = f.read()
                # print(context1)
                # print(context1)
                answer1 = qa({'question': question.rstrip('\n'), 'context': context1})
                answer2 = qa({'question': question.rstrip('\n'), 'context': context2})
                answer3 = qa({'question': question.rstrip('\n'), 'context': context3})
                answer4 = qa({'question': question.rstrip('\n'), 'context': context4})
                
                fout.write(f"Original: '{answer1['answer'].lstrip().rstrip()}' with score {answer1['score']}\n")
                fout.write(f"Neural: '{answer2['answer'].lstrip().rstrip()}' with score {answer2['score']}\n")
                fout.write(f"Clustering: '{answer4['answer'].lstrip().rstrip()}' with score {answer4['score']}\n")
                fout.write(f"NeuralCoref: '{answer3['answer'].lstrip().rstrip()}' with score {answer3['score']}\n")
                num += 1
    print('Finished')

"""
def resolve(client, text):
    ann = client.annotate(text)
    # print(dir(ann.sentence[0].token))
    copy_ann = [[token.word for token in sentence.token] for sentence in ann.sentence]
    coref_chain = ann.corefChain
    # print(coref_chain)
    coref_dict = {}

    for chain in coref_chain:
        mychain = []
        # Loop through every mention of this chain
        best_mention = chain.mention[0]
        best_punct = count_punct(ann.sentence[best_mention.sentenceIndex].token[best_mention.beginIndex:best_mention.endIndex])
        for mention in chain.mention:
            # Get the sentence in which this mention is located, and get the words which are part of this mention
            # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
            punct = count_punct(ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex])
            if moreRepresentativeThan(mention, best_mention, punct, best_punct):
                best_mention = mention
                best_punct = punct
        replace = ann.sentence[best_mention.sentenceIndex].token[best_mention.beginIndex:best_mention.endIndex]
        for mention in chain.mention:
            copy_ann[mention.sentenceIndex][mention.beginIndex] = '_'.join([token.word for token in replace])
            for index in range(mention.beginIndex + 1, mention.endIndex):
                copy_ann[mention.sentenceIndex][index] = ''
        
    new_sentence = ' '.join([' '.join([word for word in words if word != '']) for words in copy_ann])
    return new_sentence


def neuralcoref_annotate(lecture):
    with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
        context = f.read()
    context = context.split('\n\n\n')
    conts = []
    for cont in context:
        doc = nlp(cont)
        conts.append(doc._.coref_resolved)
        # print('Writing...')
    with open(f'context/Lecture_{lecture}_nc_chunk_resolved.txt', 'w+', encoding="utf-8") as fout:
        fout.write('\n\n\n'.join(conts))
    return


def cluster_annotate(lecture):
    with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
        context = f.read()
    conts = []
    count = 0
    with CoreNLPClient(
            properties={'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref', 
                        'coref.algorithm': 'clustering',
                        'parse.maxlen': 50},
            timeout=10000000,
            memory='32G') as client:
        for cont in context.split('\n\n\n'):
            print('working on chunk', count)
            cont = resolve(client, cont)
            conts.append(cont)
            count += 1
        with open(f'context/Lecture_{lecture}_clustering_resolved.txt', 'w+', encoding="utf-8") as fout:
            fout.write('\n\n\n'.join(conts))
    return

def annotate(lecture):
    with open(f'context/Lecture_{lecture}.txt', 'r', encoding="utf-8") as f:
        context = f.read()
    conts = []
    count = 0
    with CoreNLPClient(
            properties={'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref', 
                        'coref.algorithm': 'neural',
                        'parse.maxlen': 50},
            verbose=False,
            timeout=10000000,
            memory='32G') as client:
        for cont in context.split('\n\n\n'):
            print('working on chunk', count)
            cont = resolve(client, cont)
            conts.append(cont)
            count += 1
        with open(f'context/Lecture_{lecture}_resolved.txt', 'w+', encoding="utf-8") as fout:
            fout.write('\n\n\n'.join(conts))
    return


def test_annotate():
    with open(f'context/Lecture_1.txt', 'r', encoding="utf-8") as f:
        context = f.read()
    context = context.split('\n\n\n')[0]
    with CoreNLPClient(
            properties={'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref', 
                        'coref.algorithm': 'neural',
                        'parse.maxlen': 50},
            timeout=10000000,
            memory='32G') as client:
        cont = resolve(client, context)
        print(cont)
    return 
"""


def main():
    
    # test_annotate()
    """
    for lecture in range(1, 4):
        print('Annotating lecture', lecture)
        cluster_annotate(lecture)
        annotate(lecture)
    """
    """
    for model_name in ['xlnet', 'albert']:
        test_qa(model_name)
    """
    qa_bot('roberta')
    # t2t_bot()
    # test_annotate()
    


if __name__ == '__main__':
    main()