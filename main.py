"""应用入口：启动登录界面。"""
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parent
    # utf-8-sig：避免 Windows 记事本保存的 UTF-8 BOM 把变量名变成「\ufeffSILICONFLOW...」导致读不到密钥
    load_dotenv(_project_root / ".env", encoding="utf-8-sig")
    # 当前工作目录下的 .env 作为补全（override=False 时不覆盖已存在的项）
    load_dotenv(Path.cwd() / ".env", encoding="utf-8-sig")
except ImportError:
    pass

from PyQt5 import QtWidgets

from src.app.auth_windows import LoginWindow, RegistrationWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    login = LoginWindow()
    register = RegistrationWindow()
    login.register_button.clicked.connect(register.Open)
    login.show()
    sys.exit(app.exec_())
