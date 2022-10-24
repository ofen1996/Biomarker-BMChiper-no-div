import os
import sys
from PyQt5.Qt import *
from PyQt5 import QtWidgets




class FileDirBase():
    def __init__(self):
        self.cwd = os.getcwd()  # 获取当前程序文件位置
        pass

    # -------------------------------------------------------------------------------------
    # 打开文件
    # 打开目录
    # -------------------------------------------------------------------------------------
    # 打开文件
    @staticmethod
    def open_file(win, type_str='*.txt'):
        if hasattr(win, 'cwd'):
            if win.cwd is None:
                win.cwd = os.getcwd()  # 获取当前程序文件位置
        else:
            win.cwd = os.getcwd()  # 获取当前程序文件位置
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            win, '选择文件', win.cwd, f'img({type_str});;All Files (*)')

        if fileName_choose == "":
            print("\n取消选择")
            return None

        dir_choose, _ = os.path.split(fileName_choose)
        win.cwd = dir_choose

        print(f"你选择的文件为:{fileName_choose}")
        return fileName_choose
    pass
    # 打开目录
    @staticmethod
    def open_dir(win):
        if hasattr(win, 'cwd'):
            if win.cwd is None:
                win.cwd = os.getcwd()  # 获取当前程序文件位置
        else:
            win.cwd = os.getcwd()  # 获取当前程序文件位置
        dir_choose = QFileDialog.getExistingDirectory(win, "选取文件夹", win.cwd)

        if dir_choose == "":
            print("\n取消选择")
            return None

        print(f"你选择的文件夹为:{dir_choose}")
        win.cwd = dir_choose
        return dir_choose




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QMainWindow()
    window.show()
    # filename = FileDirBase.open_file(window,"*.txt")
    filename = FileDirBase.open_dir(window)
    print(f"filename_choose:{filename}")
    sys.exit(app.exec_())

