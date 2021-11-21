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
        central_widget = QWidget()
        hbox = QVBoxLayout()
        hbox.addWidget(self.lecture_dropdown)
        hbox.addWidget(self.chat_widget)
        hbox.addWidget(self.chat_input)
        
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(GUI_FONT)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())