import torch
import time
import sys
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

DEBUG = True
LECTURES = []
for filename in Path('./lectures').glob('*.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        LECTURES.append(f.read())

R1 = sr.Recognizer()
R1.pause_threshold = 1
R1.energy_threshold = 50

R2 = sr.Recognizer()
R2.pause_threshold = 1
R2.energy_threshold = 50

GUI_FONT = QFont("Arial", 12)
if not DEBUG:
    QA_MODEL = AutoModelForQuestionAnswering.from_pretrained(
        'deepset/roberta-base-squad2',
    )
    QA_MODEL.eval()
    QA_TOKENIZER = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    QA_PIPELINE = pipeline(
        "question-answering", 
        model=QA_MODEL,                
        tokenizer=QA_TOKENIZER,
        device=0,
    )


class HotWordThread(QThread):
    hotword_signal = pyqtSignal()

    def run(self):
        with sr.Microphone() as source:
            while True:
                audio = R2.listen(source)
                R2.adjust_for_ambient_noise(source)
                try:        
                    speech = R2.recognize_google(audio).lower()
                    if 'hey serious' in speech:
                        self.hotword_signal.emit()
                except Exception as e:
                    pass
                time.sleep(1)


class QuestionThread(QThread):
    question_signal = pyqtSignal(str)
        
    def run(self):
        """
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
        """
        with sr.Microphone() as source:
            print('Recognizing...')
            audio = R2.listen(source)
            R2.adjust_for_ambient_noise(source)
            try:        
                question = R2.recognize_google(audio)
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
            answer = QA_PIPELINE({
                'context': self.context, 
                'question': self.question})['answer']
            self.answer_signal.emit(answer)


class ChatWidget(QListWidget):
    class ChatWidgetItem(QWidget):
        def __init__(self, text, is_user=True, parent=None):
            super().__init__(parent)
            self.is_user = is_user
            self.label = QLabel(text)
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

        def setText(self, text):
            self.label.setText(text)


    def __init__(self, parent=None):
        super(ChatWidget, self).__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.threads = []
    
    def add_message(self, text, is_user=True):
        chat_item = self.ChatWidgetItem(text, is_user)
        item = QListWidgetItem()
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
        self.itemWidget(self.item(self.count()-1)).setText(answer)


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
        self.chat_widget.itemWidget(self.chat_widget.item(self.chat_widget.count()-1)).setText(question)
        self.setEnabled(True)
        if question == 'Not recognized':
            return
        self.chat_widget.answer(question, LECTURES[int(self.lecture_dropdown.currentText()) - 1])

    