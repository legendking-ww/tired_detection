"""应用入口：启动登录界面。"""
import sys

from PyQt5 import QtWidgets

from src.app.auth_windows import LoginWindow, RegistrationWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    login = LoginWindow()
    register = RegistrationWindow()
    login.register_button.clicked.connect(register.Open)
    login.show()
    sys.exit(app.exec_())
