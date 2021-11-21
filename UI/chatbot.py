from PyQt5 import QtCore, QtGui, QtWidgets
from widgets import *


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.resize(600, 800)
        self.chat_widget = ChatWidget()
        self.lecture_dropdown = LectureDropdown()
        self.lecture_dropdown.setFocus()
        self.chat_input = ChatInput(self.chat_widget, self.lecture_dropdown)
        self.microphone_button = AudioButton(self.chat_widget, self.lecture_dropdown)
        central_widget = QWidget()
        vbox = QVBoxLayout()
        input_hbox = QHBoxLayout()
        input_hbox.addWidget(self.chat_input)
        input_hbox.addWidget(self.microphone_button)
        vbox.addWidget(self.lecture_dropdown)
        vbox.addWidget(self.chat_widget)
        vbox.addLayout(input_hbox)
        
        central_widget.setLayout(vbox)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(GUI_FONT)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())