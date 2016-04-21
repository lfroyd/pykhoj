import sys
from PyQt4.QtGui import QApplication, QIcon
import MainWindow

def main():
    app = QApplication(sys.argv)
    # app.setOrganizationName("Qtrac Ltd.")
    # app.setOrganizationDomain("qtrac.eu")
    app.setApplicationName("Khoj 2016")
    #app.setWindowIcon(QIcon(":/icon.png"))
    window = MainWindow.Window()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()