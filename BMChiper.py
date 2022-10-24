import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000
import cv2
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
import os
from PyQt5.Qt import *

from ChipRegion import Ui_MainWindow
# from MRXSBase import MRXSBase
from ChipRegionWidget import ChipRegionWidget
from FileDirBase import FileDirBase





# 定义图片展示函数
ShowImageType = 1
def img_show(name, img):
    if ShowImageType == 0:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.imshow(img)
        plt.title(name)
        plt.show()



class ChipRegionMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ChipRegionMain, self).__init__(parent)
        self.cwd = os.getcwd()  # 获取当前程序文件位置

        # 初始化图形界面
        self.setupUi(self)
        # 对图形界面进行展示设置
        self.setupWinShow()
        pass

    def setupWinShow(self):
        self.setMinimumSize(600, 400)
        self.centralwidget.setStyleSheet('#centralwidget{background-color:#FAFBF3;}')

        self.left_scrollArea.setStyleSheet('#widget_left{background-color:#F5F9E3;}')
        self.left_scrollArea.setFixedWidth(260)
        self.toolBox.setCurrentIndex(0)
        # print(self.widget_right)

        self.widget_right = ChipRegionWidget(self.centralwidget)
        self.widget_right.setObjectName("widget_right")
        self.gridLayout_main.addWidget(self.widget_right, 0, 1, 1, 1)
        self.widget_right.setStyleSheet('#widget_right{background-color:#FF0000;}')

        self.setWindowTitle('BMChiper V1.0')
        self.setWindowIcon(QIcon('../bmk_logo.png'))  # 设置窗体标题图标

        # 设置放大
        pass
    pass



    # ------------------------------------------------
    # 手动进行芯片区域识别
    # ------------------------------------------------
    def chip_region_setup_env(self):
        self.widget_right = ChipRegionWidget(self.centralwidget)
        self.widget_right.setObjectName("widget_right")
        self.gridLayout_main.addWidget(self.widget_right, 0, 1, 1, 1)
        self.widget_right.setStyleSheet('#widget_right{background-color:#FF0000;}')
    def open_he_img_cao(self):
        if not isinstance(self.widget_right, ChipRegionWidget):
            self.chip_region_setup_env()
        filename_choose = FileDirBase.open_file(self,'*.tif *.tiff')
        if filename_choose is None:
            return
        self.widget_right.read_image(filename_choose)

        pass
    # 采集芯片区域的4个点
    # pos, 0: left_top, 1: right_top, 2: right_bottom, 3: left_bottom
    def collect_points(self, pos):
        print("pos: ", pos)
        self.widget_right.collect_points_on_ori_img(pos)
        pass
    def collect_point_left_top_cao(self):
        self.collect_points(0)
        pass
    def collect_point_right_top_cao(self):
        self.collect_points(1)
        pass
    def collect_point_right_bottom_cao(self):
        self.collect_points(2)
        pass
    def collect_point_left_bottom_cao(self):
        self.collect_points(3)
        pass
    # 保存芯片区域的4个顶点的信息
    def save_chip_region_rect_cao(self):
        rect_points = self.widget_right.min_rect_points()
        self.widget_right.warp_save_chip_region(rect_points)
        print("rect_points",rect_points)
        pass





    # -------------------------------------------------------------------------
    # 重写close方法
    # -------------------------------------------------------------------------
    # 用于提示用户保存项目
    def closeEvent(self, event):
        result = QtWidgets.QMessageBox.question(self, "程序退出确认", "退出前请保存项目，确认要退出吗？",
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if (result == QtWidgets.QMessageBox.Yes):
            event.accept()
        else:
            event.ignore()

    # -------------------------------------------------------------------------
    # File
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # 芯片区域识别
    # -------------------------------------------------------------------------
    # 全自动识别，即参数据自行进行计算
    def auto_reg_chip_region_cao(self):
        pass
        # self.autop = AutoControl(self.project_obj, run_step=1)
        # self.autop.start()
        pass





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChipRegionMain()
    window.show()
    sys.exit(app.exec_())

