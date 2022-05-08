import jsonlines
import openai
from pathlib import Path
from doc_cutter import DocCutter

ROOT_PATH = Path('lectures')

openai.organization =  # your openai organization
openai.api_key = # your api key

# print(dir(openai))

INTENT_EXAMPLE = ''
with open('train/intent.txt', 'r') as f:
    INTENT_EXAMPLE = f.read()
INTENT = lambda question:\
    f"""{INTENT_EXAMPLE.strip()}\nQuestion: {question.lower()} Label:"""

LECTURE_EXAMPLE = ''
with open('train/lecture.txt', 'r') as f:
    LECTURE_EXAMPLE = f.read()
LEC = lambda question:\
    f"""{LECTURE_EXAMPLE.strip()}\nQuery: {question.lower()} Lecture:"""

QA_EXAMPLE = ''
with open('train/qa_lecture.txt', 'r') as f:
    QA_EXAMPLE = f.read()
QA = lambda question:\
    f"""{QA_EXAMPLE.strip()}\nInput: {question.lower()} Question:"""

SEARCH_EXAMPLE = ''
with open('train/search.txt', 'r') as f:
    SEARCH_EXAMPLE = f.read()
SEARCH = lambda question:\
    f"""{SEARCH_EXAMPLE.strip()}\nQuestion: {question.lower()} Query:"""

CONTEXT_EXAMPLE = ''
with open('train/context.txt', 'r', encoding='utf-8') as f:
    CONTEXT_EXAMPLE = f.read()



def cutter(text, max_sz=1000):
    cr = DocCutter()
    return cr(text, max_sz=max_sz)

def cut_doc(filename):
    chunks = None
    with open(ROOT_PATH/filename, encoding='utf-8') as f:
        doc = f.read()
        doc = doc.replace('\n', ' ')
        chunks = cutter(doc, max_sz=1000)
    with open(ROOT_PATH/f"{filename}_chunk1", 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')

def get_files():
    files = []
    for file in openai.File.list()['data']:
        if file['purpose'] == 'answers':
            files.append(file['filename'])
    return files

def list_file():
    for file in openai.File.list()['data']:
        print(file['filename'], file['id'], file['purpose'])

def file_id(filename):
    for file in openai.File.list()['data']:
        if file['filename'] == filename:
            return file['id']
    return None

def upload_file(filename, purpose='answers'):
    response = openai.File.create(
                   file=open(ROOT_PATH/filename, encoding='utf-8'),
                   purpose=purpose,
                   )   
    # print(response)

def delete_file(filename):
    openai.File.delete(file_id(filename))

def upload_search_file():
    delete_file('Search.jsonl')
    with jsonlines.open(ROOT_PATH/'Search.jsonl', 'w') as writer:
        for file in ROOT_PATH.glob('*.jsonl'):
            if file.stem != 'Search':
                with jsonlines.open(file) as reader:
                    for obj in reader:
                        writer.write(obj)
    # print('finished')
    upload_file('Search.jsonl')
        

def chunker(filename):
    lines = []
    with open(ROOT_PATH/filename, encoding='utf-8') as f:
        """
        ls = []
        last_line = ''
        for line in f:
            line = line.strip()
            if line.endswith('.'):
                ls.append(last_line + line)
                last_line = ''
            else:
                last_line += line
        if last_line not in ls:
            ls.append(last_line)

        for line in ls:
            line = line.strip()
            if len(line.split()) < 5:
                continue
            if len(line.split()) > 2000:
                sents = line.split('. ')
                new_line = []
                length = 10000
                index = -1
                for sent in sents:
                    if length + len(sent) < 2000:
                        new_line[index].append(sent)
                        length += len(sent)
                    else:
                        new_line.append([])
                        index += 1
                        new_line[index].append(sent)
                        length = len(sent)
                lines.extend(new_line)
            else:
                lines.append([line])
        """
        sents = []
        for lines in f:
            lines = lines.strip()
            sents.extend(lines.split('. '))
        lines = []
        line = ''
        for sent in sents:
            if len(line.split()) + len(sent.split()) < 1000:
                line += sent + '. '
            else:
                lines.append(line)
                line = sent + '. '
        if line not in lines:
            lines.append(line)
    with open(ROOT_PATH/f"{filename}_chunk", 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '.')
            f.write('\n')

def txt2jsonl(filename):
    with open(ROOT_PATH/filename, encoding='utf-8') as f:
        with jsonlines.open(ROOT_PATH/f"{filename.split('.')[0]}.jsonl", 'w') as writer:
            for line in f:
                writer.write({"text": line})

def txt_upload(filename):
    cut_doc(filename)
    txt2jsonl(filename + '_chunk1')
    upload_file(filename.split('.')[0] + '.jsonl')
    # upload_search_file()

def answer_question(question, filename=None, max_tokens=10):
    if filename is None:
        try:
            response = openai.Completion.create(
                                engine="davinci",
                                prompt=f'Question: {question} Answer:',
                                temperature=0, 
                                max_tokens=max_tokens
                                )
            return response['choices'][0]['text'].split('Question')[0].strip()
        except:
            return None
    try:
        response = openai.Answer.create(
                        search_model="ada",
                        model="curie",
                        question=question,
                        file=file_id(filename),
                        examples_context=CONTEXT_EXAMPLE,
                        examples=[["Where is the capital of Germany?", "The capital of Germany is Berlin."], 
                                  ["Where is Berlin in Germany?", "Berlin is in Northeastern Germany."], 
                                  ["What task was the gpt-3 model trained on?", "The gpt-3 model was trained to predict the next word of an input sequence."],
                                  ["Where is the hidden layer?", "The hidden layer is after the input layer."],
                                  ["How many parameters are there in gpt-3", "There are 175 billion parameters in gpt-3."]],
                        max_tokens=max_tokens,
                        stop=["\n", "<|endoftext|>"],
                        )
        # print(response)
        return response['answers'][0].strip()
    except Exception as e:
        print(e)
        return None

def summarize(text, max_tokens=10):
    try:
        response = openai.Completion.create(
                        engine="davinci",
                        prompt=f'Here\'s what the speaker said, "{text}" The main point was that',
                        temperature=0.05, 
                        max_tokens=max_tokens
                        )
        # print(response)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        # print(e)
        return None

def search(text):
    query = query_extractor(text)
    print(query)
    if query is None:
        return None
    try:
        response = openai.Engine('ada').search(
                            search_model="ada",
                            query=query,
                            file=file_id('Search.jsonl'),
                            max_rerank=2,
                            )
        # print(response)
        return response['data'][0]['text']
    except:
        return None

def intent_classifier(text):
    try:
        response = openai.Completion.create(
                            engine="davinci",
                            prompt=INTENT(text),
                            temperature=0, 
                            max_tokens=1
                            )
        return response['choices'][0]['text'].strip()
    except:
        return 'question'

def question_extraction(text):
    try:
        response = openai.Completion.create(
                            engine="davinci",
                            prompt=QA(text),
                            temperature=0, 
                            max_tokens=10
                            )
        return response['choices'][0]['text'].split('Input')[0].strip()
    except:
        return None

def lecture_classifier(text):
    try:
        response = openai.Completion.create(
                            engine="davinci",
                            prompt=LEC(text),
                            temperature=0.05, 
                            max_tokens=1
                            )
        # print(response)
        return response['choices'][0]['text'].strip()
    except:
        return None

def query_extractor(text):
    try:
        response = openai.Completion.create(
                            engine="davinci",
                            prompt=SEARCH(text),
                            temperature=0, 
                            max_tokens=10
                            )
        return response['choices'][0]['text'].split('Question')[0].strip()
    except:
        return None

def paraphrase(text, max_tokens=10):
    try:
        response = openai.Completion.create(
                        engine="davinci",
                        prompt=f'{text}\n To sum up, ',
                        temperature=0.1, 
                        max_tokens=max_tokens
                        )
        # print(response)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        # print(e)
        return None

def main():
    print(upload_file('Lecture_1.jsonl'))

if __name__ == '__main__':
    delete_file('Lecture_7.jsonl')
    list_file()