import torch
import time
import sys
import json
import os
import pyaudio
import wave
import speech_recognition as sr
# import dialogflow
from pathlib import Path
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from tempfile import TemporaryDirectory
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Xianglong\\Desktop\\UMich\\Courses\\EECS 498 Conversation AI\\Project Sirious\\Sirious\\UI\\cert\\key.json'
# os.environ['GCLOUD_PROJECT'] = 'sirious'

DEBUG = True
LECTURES = []
for filename in Path('./lectures').glob('*.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        LECTURES.append(f.read())

R1 = sr.Recognizer()
R1.pause_threshold = 0.5
R1.energy_threshold = 300
# R1.dynamic_energy_threshold = True

R2 = sr.Recognizer()
R2.pause_threshold = 1
R2.energy_threshold = 300
# R1.dynamic_energy_threshold = True

GCLOUD_CREDENTIALS = json.dumps(json.load(open('cert/key.json')))

GUI_FONT = QFont("Arial", 12)
if not DEBUG:
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


"""
def bigbird_pegasus(text):
    inputs = SUM_TOKENIZER([text], max_length=4096, return_tensors='pt', truncation=True)
    # return inputs['input_ids']
    # print(text[:1000])
    # summary_ids = model.generate(inputs['input_ids'], num_beams=8, max_length=512, early_stopping=True)
    summary_ids = SUM_MODEL.generate(inputs['input_ids'], num_beams=8, max_length=512)
    # return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return SUM_TOKENIZER.batch_decode(summary_ids)
"""

class HotWordThread(QThread):
    hotword_signal = pyqtSignal()

    def run(self):
        with sr.Microphone() as source:
            while True:
                audio = R1.listen(source)
                R1.adjust_for_ambient_noise(source)
                try:        
                    speech = R1.recognize_google_cloud(audio).lower()
                    if 'hey serious' in speech:
                        self.hotword_signal.emit()
                except Exception as e:
                    pass
                time.sleep(1)


class QuestionThread(QThread):
    question_signal = pyqtSignal(str)

    def run(self):
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
    answer_signal = pyqtSignal(str)
    def __init__(self, question, context, parent=None):
        super().__init__(parent)
        self.question = question
        self.context = context

    def run(self):
        with torch.no_grad():
            if "summarize" in self.question.lower():
                for block in self.context.split('\n\n'):
                    if len(block) > 0:
                        for summary in SUM_PIPELINE(block, min_length=5, max_length=20):
                            self.answer_signal.emit(summary['summary_text'])
                    break
            else:
                answer = QA_PIPELINE({
                    'context': self.context, 
                    'question': self.question})['answer']
                self.answer_signal.emit(answer)


class ChatWidget(QListWidget):
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

    def __init__(self, parent=None):
        super(ChatWidget, self).__init__(parent)
        self.setMinimumSize(600, 800)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.threads = []
    
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

    def answer(self, question, context):
        self.add_message('...', is_user=False)
        if not DEBUG:
            answer_thread = AnswerThread(question, context)
            answer_thread.answer_signal.connect(self.on_answer_received)
            answer_thread.start()
            self.threads.append(answer_thread)
        # self.cur_answer_item = self.chat_widget.item(self.chat_widget.count()-1)
        """
        with torch.no_grad():
            answer = PIPELINE({'question': text.rstrip('\n'), 'context': context})
        self.chat_widget.add_message(answer['answer'], is_user=False)
        """
    
    def on_answer_received(self, answer):
        if self.itemWidget(self.item(self.count()-1)).getText() == '...':
            self.set_message(answer, self.item(self.count()-1), is_user=False)
        else:
            self.add_message(answer, is_user=False)


class LectureDropdown(QComboBox):
    def __init__(self, parent=None):
        super(LectureDropdown, self).__init__(parent)
        self.addItems([str(i+1) for i in range(len(LECTURES))])
        self.setCurrentIndex(0)


class ChatInput(QLineEdit):
    def __init__(self, chat_widget, lecture_dropdown, parent=None):
        super(ChatInput, self).__init__(parent)
        self.setPlaceholderText("Type your message here...")
        self.chat_widget = chat_widget
        self.lecture_dropdown = lecture_dropdown
        self.returnPressed.connect(self.on_return_pressed)


    def on_return_pressed(self):
        print('return presseed')
        text = self.text()
        if not text:
            self.chat_widget.add_message('Please enter a question.', is_user=False)
            return
        if text == 'exit':
            sys.exit()
        self.chat_widget.add_message(text, is_user=True)
        context = LECTURES[int(self.lecture_dropdown.currentText()) - 1]
        self.chat_widget.answer(text.rstrip('\n'), context)
        self.clear()
    

class AudioButton(QPushButton):
    def __init__(self, chat_widget, lecture_dropdown, parent=None):
        super(AudioButton, self).__init__(parent)
        self.setIcon(QIcon('icon/microphone.png'))
        self.setIconSize(QSize(28, 28))
        self.chat_widget = chat_widget
        self.clicked.connect(self.on_clicked)
        self.question_thread = None
        self.lecture_dropdown = lecture_dropdown
        self.hotword_thread = HotWordThread()
        self.hotword_thread.hotword_signal.connect(self.on_clicked)
        self.hotword_thread.start()


    def on_clicked(self):
        self.chat_widget.add_message('...', is_user=True)
        self.question_thread = QuestionThread()
        self.question_thread.question_signal.connect(self.on_question_received)
        self.question_thread.start()
        self.setEnabled(False)

    def on_question_received(self, question):
        self.chat_widget.set_message(question, self.chat_widget.item(self.chat_widget.count()-1), is_user=True)
        self.setEnabled(True)
        if question == 'Not recognized':
            return
        self.chat_widget.answer(question, LECTURES[int(self.lecture_dropdown.currentText()) - 1])

    