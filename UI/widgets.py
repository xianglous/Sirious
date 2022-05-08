import torch
import time
import sys
import json
import os
import pyaudio
import wave
import jsonlines
import pyttsx3 
import speech_recognition as sr
# import dialogflow
from string import punctuation
from pathlib import Path
from collections import deque
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtTextToSpeech import QTextToSpeech
from tempfile import TemporaryDirectory
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from gpt3 import get_files, answer_question, summarize, paraphrase, intent_classifier, lecture_classifier, question_extraction, search, cutter, txt_upload
from transcript import transcribe, audio2flac, video2audio, upload, delete


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Xianglong\\Desktop\\UMich\\Courses\\EECS 498 Conversation AI\\Project Sirious\\Sirious\\UI\\cert\\key.json'
os.environ['GCLOUD_PROJECT'] = 'sirious'

USE_MODEL = False
USE_HOTWORD = False
LECTURES = []
for filename in Path('./lectures').glob('*.jsonl'):
    if not filename.stem.startswith('Lecture_'):
        continue
    with jsonlines.open(filename) as f:
        LECTURES.append([])
        for read_obj in f:
            LECTURES[-1].append(read_obj['text'])


SPEAKER = pyttsx3.init()

for voice in SPEAKER.getProperty('voices'):
    if 'EN-US' in voice.id:
        SPEAKER.setProperty('voice', voice.id)
        break

R1 = sr.Recognizer()
R1.pause_threshold = 0.5
R1.energy_threshold = 300
# R1.dynamic_energy_threshold = True

R2 = sr.Recognizer()
R2.pause_threshold = 1
R2.energy_threshold = 500
R2.dynamic_energy_threshold = False

# GCLOUD_CREDENTIALS = json.dumps(json.load(open('cert/key.json')))

GUI_FONT = QFont("Arial", 12)
if USE_MODEL:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    QA_MODEL = AutoModelForQuestionAnswering.from_pretrained(
        'deepset/roberta-base-squad2',
    )
    QA_MODEL.load_state_dict(torch.load('model/QA_checkpoint/pytorch_model.bin'))
    # QA_MODEL.to(DEVICE)
    QA_MODEL.eval()
    QA_TOKENIZER = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    QA_PIPELINE = pipeline(
        "question-answering", 
        model=QA_MODEL,
        tokenizer=QA_TOKENIZER,
        device=0,
    )
    SUM_MODEL = 't5-base'
    SUM_TOKENIZER = AutoTokenizer.from_pretrained(SUM_MODEL, model_max_len=512)
    # SUM_TOKENIZER = PegasusTokenizer.from_pretrained('google/bigbird-pegasus-large-arxiv')
    SUM_PIPELINE = pipeline(
        "summarization",
        device=0,
    )


class HotWordThread(QThread):
    hotword_signal = pyqtSignal()
    def __init__(self, parent=None):
        super(HotWordThread, self).__init__(parent)
        self.listening = True

    def run(self):
        while True:
            if self.listening:
                with sr.Microphone() as source:
                    self.listen(source)
            # time.sleep(1)
    
    def listen(self, source):
        while self.listening:
            audio = R1.listen(source)
            R1.adjust_for_ambient_noise(source)
            try:        
                speech = R1.recognize_google_cloud(audio).lower()
                if 'hey serious' in speech:
                    self.hotword_signal.emit()
            except Exception as e:
                print(repr(e))
                pass
            time.sleep(1)
    
    def toggle_pause(self, pause):
        self.listening = (not pause)


class QuestionThread(QThread):
    question_signal = pyqtSignal(str)

    def __init__(self, hotword_thread, parent=None):
        super(QuestionThread, self).__init__(parent)            
        self.hotword_thread = hotword_thread

    def run(self):
        if self.hotword_thread is None == False:
            self.hotword_thread.toggle_pause(True)
        with sr.Microphone() as source:
            print('Recognizing...')
            audio = R2.listen(source)
            R2.adjust_for_ambient_noise(source)
            try:        
                question = R2.recognize_google_cloud(audio)
                print(question)
                self.question_signal.emit(question)
            except Exception as e:
                print(e)
                self.question_signal.emit('Not recognized')


class AnswerThread(QThread):
    answer_signal = pyqtSignal(str, int)
    def __init__(self, question, index, parent=None):
        super().__init__(parent)
        self.question = question
        # self.lecture_num = lecture_num
        self.index = index
        self.running = True

    def run(self):
        intent = intent_classifier(self.question)
        print(intent)
        try:
            intent = int(intent)
        except Exception as e:
            intent = 0
        if intent not in [0, 1, 2]:
            intent = 0
        if USE_MODEL:
            # context = LECTURES[self.lecture_num - 1]
            with torch.no_grad():
                if "summarize" in self.question.lower():
                    for block in context:
                        if len(block) > 0:
                            for summary in SUM_PIPELINE(block, min_length=5, max_length=20):
                                self.answer_signal.emit(summary['summary_text'], -1)
                        break
                else:
                    answer = QA_PIPELINE({
                        'context': '\n'.join(context), 
                        'question': self.question})['answer']
                    self.answer_signal.emit(answer, -1)
        else:
            # context = LECTURES[self.lecture_num - 1]
            if "summarize" in self.question.lower() or intent == 1:
                def summary(chunks):
                    summs = ''
                    sents = set()
                    for block in chunks:
                        if len(block.split()) < 50:
                            continue
                        summ = summarize(block, 50)
                        # print(summ)
                        if summ is None:
                            continue
                        summ = summ.strip() + ' '
                        new_summ, sents = self.clean_text(summ, sents)
                        if len(new_summ) == 0:
                            continue
                        new_summ = '. '.join(new_summ) + '. '
                        summs += new_summ
                    return summs
                lec_num = lecture_classifier(self.question)
                try:
                    lec_num = int(lec_num) - 1
                except Exception as e:
                    lec_num = -1
                if lec_num not in range(len(LECTURES)):
                    self.answer_signal.emit('Sorry, this lecture doesn\'t exist', self.index)
                    return
                summs = summary(LECTURES[lec_num])
                while len(summs.split()) > 250:
                    sums = cutter(summs)
                    summs = summary(sums)
                self.answer_signal.emit(summs, self.index)
            elif intent == 0:
                lecture = lecture_classifier(self.question)
                question = question_extraction(self.question).lower()
                print(lecture, question)
                try:
                    # lecture = lecture.strip()
                    lecture = int(lecture) - 1
                except:
                    lecture = -1
                old_lec = lecture
                if lecture not in range(len(LECTURES)):
                    lec = self.get_lecture(question)
                    # print(lec)
                    if lec == -1:
                        answer = answer_question(question)
                        if answer is None or answer.strip() == '':
                            self.answer_signal.emit('Sorry, can you say it again?', self.index)
                            return
                        answer = answer.strip() + ' '
                        answer, _ = self.clean_text(answer, set())
                        answer = '. '.join(answer) + '. '
                        self.answer_signal.emit(f'This doesn\'t seem to appear anywhere in the lecture, but I think the answer is {answer}', self.index)
                        return
                    else:
                        lecture = lec - 1
                answer = answer_question(question, f"Lecture_{lecture + 1}.jsonl", 50)
                print(question, lecture, answer)
                if answer is None or answer.strip() == '':
                    self.answer_signal.emit('Sorry, can you rephrease your question?', self.index)
                    return
                answer = answer.strip() + ' '
                answer, _ = self.clean_text(answer, set())
                answer = '. '.join(answer)
                if old_lec == lecture:
                    self.answer_signal.emit(f'{answer}', self.index)
                else:
                    self.answer_signal.emit(f'{answer} - Lecture {lecture + 1}', self.index)
            else:
                lec = self.get_lecture(self.question)
                if lec == -1:
                    self.answer_signal.emit('Sorry, this lecture doesn\'t exist', self.index)
                    return
                self.answer_signal.emit(f"It is in Lecture {lec}", self.index)
    
    def clean_text(self, text, sents):
        if '. ' in text and text[-2:] != '. ':
            text = text.split('. ')[:-1]
        else:
            text = text.split('. ')
        new_text = []
        for s in text:
            s = s.strip().strip(punctuation)
            if len(s) == 0:
                continue 
            if s.lower() not in sents:
                sents.add(s.lower())
                new_text.append(s.capitalize())
        return new_text, sents

    def get_lecture(self, text):
        document = search(text)
        if document is None:
            return -1
        lec = 1        
        for lecture in LECTURES:
            if document in lecture:
                # self.answer_signal.emit(f'It is in lecture {lec}', self.index)
                return lec
            lec += 1
        return -1


class ChatWidget(QListWidget):
    answer_signal = pyqtSignal(str)
    class ChatWidgetItem(QWidget):
        def __init__(self, text, is_user=True, parent=None):
            super().__init__(parent)
            self.is_user = is_user
            self.label = QLabel()
            # self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.label.setWordWrap(True)
            font_metric = QFontMetrics(self.label.font())
            width = font_metric.width(text)
            if width < 400:
                self.label.setMinimumWidth(width + 20)
            else:
                self.label.setMinimumWidth(400)
            self.label.setText(text)
            hbox = QHBoxLayout()
            if is_user:
                self.label.setStyleSheet(" \
                    background-color: #376df5; \
                    color: white; \
                    padding: 5px; \
                    border-radius: 10px; \
                    ")
                hbox.addStretch(1)
                hbox.addWidget(self.label)
            else:
                self.label.setStyleSheet(" \
                    background-color: #bbbbbb; \
                    color: black; \
                    padding: 5px; \
                    border-radius: 10px; \
                    ")
                hbox.addWidget(self.label)
                hbox.addStretch(1)
            self.setLayout(hbox)

        def getText(self):
            return self.label.text()
        
        def addText(self, text):
            self.label.setText(self.label.text() + text)

    def __init__(self, parent=None):
        super(ChatWidget, self).__init__(parent)
        self.setMinimumSize(1000, 700)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.threads = []
        self.responses = deque()
        self.speaker = QTextToSpeech()
        self.speaker.setLocale(QLocale('en-US'))
        self.speaker.setRate(0.2)
        self.speaker.stateChanged.connect(self.on_state_changed)
        
    
    def add_message(self, text, is_user=True):
        chat_item = self.ChatWidgetItem(text, is_user, self)
        item = QListWidgetItem()
        item.setSizeHint(chat_item.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, chat_item)
        self.scrollToItem(item)
    
    def set_message(self, text, item, is_user=True):
        chat_item = self.ChatWidgetItem(text, is_user, self)
        item.setSizeHint(chat_item.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, chat_item)
        self.scrollToItem(item)

    def answer(self, question):
        self.add_message('...', is_user=False)
        # if not USE_MODEL:
        answer_thread = AnswerThread(question, self.count()-1)
        answer_thread.answer_signal.connect(self.on_answer_received)
        answer_thread.start()
        self.threads.append(answer_thread)
        # self.cur_answer_item = self.chat_widget.item(self.chat_widget.count()-1)
        """
        with torch.no_grad():
            answer = PIPELINE({'question': text.rstrip('\n'), 'context': context})
        self.chat_widget.add_message(answer['answer'], is_user=False)
        """

    def on_answer_received(self, answer, index):
        if index >= 0 and self.itemWidget(self.item(index)).getText() == '...':
            self.set_message(answer, self.item(index), is_user=False)
        else:
            self.add_message(answer, is_user=False)
        # print(self.speaker.state(), self.responses)
        if self.speaker.state() == QTextToSpeech.Ready:
            self.speaker.say(answer)
        else:
            self.responses.appendleft(answer)
        # self.speaker.say(answer)
    
    def on_state_changed(self, state):
        # print(state, self.responses)
        if state == QTextToSpeech.Ready:
            # print('yes', len(self.responses))
            if len(self.responses) > 0:
                # print('speak')
                self.speaker.say(self.responses[-1])
                self.responses.pop()
            


class LectureDropdown(QComboBox):
    def __init__(self, parent=None):
        super(LectureDropdown, self).__init__(parent)
        self.addItems([str(i+1) for i in range(len(LECTURES))])
        self.setCurrentIndex(0)


class ChatInput(QLineEdit):
    def __init__(self, chat_widget, parent=None):
        super(ChatInput, self).__init__(parent)
        self.setPlaceholderText("Type your message here...")
        self.chat_widget = chat_widget
        # self.lecture_dropdown = lecture_dropdown
        self.returnPressed.connect(self.on_return_pressed)


    def on_return_pressed(self):
        # print('return presseed')
        text = self.text()
        if not text:
            self.chat_widget.add_message('Please enter a question.', is_user=False)
            return
        if text == 'exit':
            sys.exit()
        self.chat_widget.add_message(text, is_user=True)
        # context = LECTURES[int(self.lecture_dropdown.currentText()) - 1]
        self.chat_widget.answer(text.rstrip('\n'))
        self.clear()
    

class AudioButton(QPushButton):
    def __init__(self, chat_widget, parent=None):
        super(AudioButton, self).__init__(parent)
        self.setIcon(QIcon('icon/microphone.png'))
        self.setIconSize(QSize(28, 28))
        self.chat_widget = chat_widget
        self.clicked.connect(self.on_clicked)
        self.question_thread = None
        # self.lecture_dropdown = lecture_dropdown
        self.hotword_thread = None
        if USE_HOTWORD:
            self.hotword_thread = HotWordThread()
            self.hotword_thread.hotword_signal.connect(self.on_clicked)
            self.hotword_thread.start()


    def on_clicked(self):
        self.chat_widget.add_message('...', is_user=True)
        self.question_thread = QuestionThread(self.hotword_thread)
        self.question_thread.question_signal.connect(self.on_question_received)
        self.question_thread.start()
        self.setEnabled(False)

    def on_question_received(self, question):
        self.chat_widget.set_message(question, self.chat_widget.item(self.chat_widget.count()-1), is_user=True)
        self.setEnabled(True)
        if question == 'Not recognized':
            return
        self.chat_widget.answer(question)
        if USE_HOTWORD:
            self.hotword_thread.toggle_pause(False)


class LectureList(QListWidget):
    def __init__(self, parent=None):
        super(LectureList, self).__init__(parent)
        self.setMinimumSize(200, 700)
        for filename in sorted(get_files()):
            self.addItem(filename.split('.')[0].replace('_', ' '))
        self.upload_threads = []
        
    def addLecture(self, file_path):
        item = QListWidgetItem(f"Lecture {self.count() + 1}")
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.addItem(item)
        upload_thread = UploadThread(str(file_path), self, self.count()-1)
        upload_thread.upload_finished_signal.connect(self.on_upload_finished)
        upload_thread.start()
        self.upload_threads.append(upload_thread)

    
    def setText(self, index, text):
        self.item(index).setText(text)
    
    def on_upload_finished(self, index):
        QMessageBox.information(self, 'Upload finished', 'Upload finished.')


class UploadThread(QThread):
    upload_finished_signal = pyqtSignal(int)
    def __init__(self, file_path, lecture_list, index, parent=None):
        super(UploadThread, self).__init__(parent)
        self.file_path = file_path
        self.lecture_list = lecture_list
        self.index = index
    
    def run(self):
        exe = self.file_path.split('.')[-1]
        self.lecture_list.setText(self.index, 'Uploading...')
        with TemporaryDirectory() as dir:
            if exe != 'flac':
                if exe in ['mp3', 'wav']:
                    audio2flac(self.file_path, dir)
                else:
                    video2audio(self.file_path, dir)
                self.file_path = Path(dir)/'output.flac'
            # self.lecture_list.setText(self.index, 'Uploading to GCS...')
            gcs_uri = "gs://" + upload(self.file_path, f"Lecture_{self.index + 1}.flac").split('https://storage.googleapis.com/')[-1]
            # self.lecture_list.setText(self.index, f'Transcribing...')
            transcribe(gcs_uri, f"Lecture_{self.index + 1}.txt")
            # self.lecture_list.setText(self.index, 'Uploading to OpenAI...')
            txt_upload(f"Lecture_{self.index + 1}.txt")
            self.lecture_list.setText(self.index, f'Lecture {self.index + 1}')
            self.upload_finished_signal.emit(self.index)


class LectureUploadButton(QPushButton):
    def __init__(self, lecture_list, parent=None):
        super(LectureUploadButton, self).__init__(parent)
        self.setText('+')
        # self.chat_widget = chat_widget
        self.lecture_list = lecture_list
        self.clicked.connect(self.on_clicked)
    
    def on_clicked(self):
        dir, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', 'Files (*.wav, *.mp3, *.flac, *.mp4)')
        if dir == '':
            return
        # print(dir)
        self.lecture_list.addLecture(Path(dir))
