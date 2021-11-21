import torch
import time
import sys
# import dialogflow
from pathlib import Path
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


LECTURES = []
for filename in Path('./lectures').glob('*.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        LECTURES.append(f.read())

GUI_FONT = QFont("Arial", 12)

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
    
    def add_message(self, text, is_user=True):
        chat_item = self.ChatWidgetItem(text, is_user)
        item = QListWidgetItem()
        item.setSizeHint(chat_item.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, chat_item)
        self.scrollToItem(item)


class LectureDropdown(QComboBox):
    def __init__(self, parent=None):
        super(LectureDropdown, self).__init__(parent)
        list_items = ['Please select a lecture']
        list_items.extend([str(i+1) for i in range(len(LECTURES))])
        self.addItems(list_items)
        self.setCurrentIndex(0)


class ChatInput(QLineEdit):
    def __init__(self, chat_widget, lecture_dropdown, parent=None):
        super(ChatInput, self).__init__(parent)
        self.setPlaceholderText("Type your message here...")
        self.chat_widget = chat_widget
        self.lecture_dropdown = lecture_dropdown
        self.returnPressed.connect(self.on_return_pressed)
        self.threads = []

    def on_answer_received(self, answer):
        self.chat_widget.itemWidget(self.chat_widget.item(self.chat_widget.count()-1)).setText(answer)

    @pyqtSlot()
    def on_return_pressed(self):
        text = self.text()
        if not text:
            self.chat_widget.add_message('Please enter a question.', is_user=False)
            return
        if text == 'exit':
            sys.exit()
        try:
            context = LECTURES[int(self.lecture_dropdown.currentText()) - 1]
        except:
            self.chat_widget.add_message(text, is_user=True)
            self.chat_widget.add_message('Please choose a lecture.', is_user=False)
            self.clear()
            return
        self.chat_widget.add_message(text, is_user=True)
        self.chat_widget.add_message('...', is_user=False)
        answer_thread = AnswerThread(text.rstrip('\n'), context)
        answer_thread.answer_signal.connect(self.on_answer_received)
        answer_thread.start()
        self.threads.append(answer_thread)
        # self.cur_answer_item = self.chat_widget.item(self.chat_widget.count()-1)
        """
        with torch.no_grad():
            answer = PIPELINE({'question': text.rstrip('\n'), 'context': context})
        self.chat_widget.add_message(answer['answer'], is_user=False)
        """
        self.clear()
        
    