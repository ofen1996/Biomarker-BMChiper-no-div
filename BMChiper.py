import sys
import traceback

import numpy as np
import tifffile
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000
import cv2
import matplotlib.pyplot as plt

import match_imgs

from PyQt5 import QtWidgets
import os
from PyQt5.Qt import *
from need.ChipRegion import Ui_MainWindow
# from MRXSBase import MRXSBase
from need.ChipRegionWidget import ChipRegionWidget
from need.FileDirBase import FileDirBase
from stitch_pic import StitchImg

from need.CorrectWholeImg import correct_whole_img

from need.config import conf





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
        # self.cwd = os.getcwd()  # 获取当前程序文件位置
        if os.path.exists(conf.conf.get("default", "base_dir")):
            self.cwd = conf.conf.get("default", "base_dir")
        else:
            self.cwd = None

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

        self.setWindowTitle('BMChiper V2.1')
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
        filename_choose = FileDirBase.open_file(self,'*.tif *.tiff *.mrxs')
        if filename_choose is None:
            return
        if filename_choose.endswith(".mrxs"):
            self.stitch_chip.setEnabled(True)
            self.new_stitch_channel.setEnabled(True)
        else:
            self.stitch_chip.setEnabled(False)
            self.new_stitch_channel.setEnabled(False)
        self.widget_right.read_image(filename_choose)

        # 记忆选择的路径
        base_dir = os.path.split(filename_choose)[0]
        conf.reload()
        conf.conf.set("default", "base_dir", base_dir)
        with open(conf.ini_path, 'w') as f:
            conf.conf.write(f)
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
        tif_file = self.widget_right.warp_save_chip_region(rect_points)
        print("rect_points", rect_points)
        self.widget_right.image_filename = tif_file
        self.widget_right.read_image(tif_file)

        pass

    # 缝合并裁剪图像
    def stitch_chip_cao(self):
        conf.reload()  # 重新读取配置文件，以支持热修改
        if [-1, -1] in self.widget_right.draw_argvs['rect_points']:
            print("请先选择4个点")
            return
        try:
            corners = np.asarray(self.widget_right.draw_argvs['rect_points'])
            # camera_resolutions = [(2448, 2048), (2048, 2048)]
            # camera_resolution = camera_resolutions[self.comboBox_camera_type.currentIndex()]
            # print("camera_resolution:" + str(camera_resolution))
            st_img = StitchImg(self.widget_right.image_filename, corners * 4, self.widget_right.channel_show)
            new_corners, level_2_path = st_img.cut_and_stitch()

            # level_2_path = "./test.tif"
            # tifffile.imwrite(level_2_path, np.asarray(self.widget_right.draw_argvs['zoom_imgs'][-1])[:, :, :3], compression="jpeg")

            self.widget_right.image_filename = level_2_path
            self.widget_right.read_image(level_2_path)
        except Exception as e:
            print(traceback.format_exc(), e)
            return

        pass


    def correct_whole_cao(self):
        print("Start correct whole img")
        conf.reload()  # 重新读取配置文件，以支持热修改
        if self.widget_right.image_filename is not None:
            detect_channel = None
            try:
                if self.widget_right.channel_show != 0:
                    detect_channel = self.widget_right.channel_show - 1
                correct_whole_img(self.widget_right.image_filename, detect_channel=detect_channel)
            except Exception as e:
                print(traceback.format_exc(), e)
                return
        else:
            print("No img Path")
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

    def choose_channel_show_cao(self):
        pass
        print(self.comboBox.currentIndex())
        self.widget_right.channel_show = self.comboBox.currentIndex()

    def new_stitch_channel_cao(self):
        conf.reload()  # 重新读取配置文件，以支持热修改
        if [-1, -1] in self.widget_right.draw_argvs['rect_points']:
            print("请先选择4个点")
            return
        try:
            corners = np.asarray(self.widget_right.draw_argvs['rect_points']) * 4
            print(corners)

            # 改变 stitch_channal
            if self.widget_right.channel_show != 0:
                stitch_channel = self.widget_right.channel_show - 1
                conf.conf.set("match-imgs", "stitch_channal", str(stitch_channel))
                with open(conf.ini_path, "w") as conf_ini:
                    conf.conf.write(conf_ini)

            stitch_path = match_imgs.cut_and_stitch(self.widget_right.image_filename, corners.tolist())
            # camera_resolutions = [(2448, 2048), (2048, 2048)]
            # camera_resolution = camera_resolutions[self.comboBox_camera_type.currentIndex()]
            # print("camera_resolution:" + str(camera_resolution))
            # st_img = StitchImg(self.widget_right.image_filename, corners * 4, self.widget_right.channel_show)
            # new_corners, level_2_path = st_img.cut_and_stitch()

            # level_2_path = "./test.tif"
            # tifffile.imwrite(level_2_path, np.asarray(self.widget_right.draw_argvs['zoom_imgs'][-1])[:, :, :3], compression="jpeg")

            self.widget_right.image_filename = stitch_path
            self.widget_right.read_image(stitch_path)
        except Exception as e:
            print(traceback.format_exc(), e)
            raise
            return

        pass





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChipRegionMain()
    window.show()
    sys.exit(app.exec_())

