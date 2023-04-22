import sys
from PySide6.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget

# create the application
app = QApplication(sys.argv)

# create a PySide6 widget
widget = QWidget()


# show the PySide6 widget
widget.show()

# enter the event loop
sys.exit(app.exec())