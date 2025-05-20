import sys
import threading
import time
import traceback
import copy
from threading import Thread
import numpy as np
# import tifffile
from PIL import Image

from need.ofen_tool import load_json

Image.MAX_IMAGE_PIXELS = 3000000000
import cv2
import matplotlib.pyplot as plt

import match_imgs
import match_imgs_ome

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

import builtins

import logging

# 为 custom_print 创建独立的日志记录器
custom_logger = logging.getLogger('custom_print')
custom_logger.setLevel(logging.INFO)
log_file = os.path.splitext(__file__)[0] + "_custom.log"  # 独立的日志文件，例如 script_custom.log
custom_handler = logging.FileHandler(log_file)
custom_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
custom_logger.addHandler(custom_handler)
custom_logger.propagate = False  # 防止日志传播到其他 handler

# 可选：配置主日志（用于其他模块的日志）
logging.basicConfig(
    filename=os.path.splitext(__file__)[0] + "_main.log",  # 其他日志写入 script_main.log
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)
main_logger = logging.getLogger(__name__)

# 禁用 pyvips 的调试日志（可选，防止其日志干扰）
logging.getLogger('pyvips').setLevel(logging.WARNING)

# 保存原始 print 函数
_original_print = builtins.print


def custom_print(*args, **kwargs):
    """自定义 print 函数，仅记录用户指定的消息"""
    try:
        # 将参数转换为字符串
        message = " ".join(str(arg) for arg in args)

        # 调用原始 print
        _original_print(*args, **kwargs)

        # 使用独立的 custom_print 日志记录器
        custom_logger.info(message)
    except Exception as e:
        _original_print(f"Logging failed: {e}", **kwargs)


# 替换内置 print 函数
builtins.print = custom_print


def show_wait_dialog(func):
    def wrapper(self, *args, **kwargs):
        # Create and show the wait dialog
        wait_msg = QMessageBox(self)
        wait_msg.setWindowTitle("请稍候")
        wait_msg.setText("执行中，请稍后...")
        wait_msg.setStandardButtons(QMessageBox.NoButton)  # No buttons
        wait_msg.setModal(True)  # Modal dialog
        wait_msg.show()

        # Force UI update
        QApplication.processEvents()

        # Variable to store the function result
        result = [None]
        exception = [None]

        # Define the target function to run in the thread
        def run_target():
            try:
                result[0] = func(self, *args, **kwargs)
            except Exception as e:
                exception[0] = e

        # Start the function in a separate thread
        thread = Thread(target=run_target)
        thread.start()

        # Keep the dialog open while the thread is running
        while thread.is_alive():
            QApplication.processEvents()  # Keep UI responsive
            wait_msg.repaint()  # Ensure dialog stays visible

        # Close the dialog
        wait_msg.done(0)
        QApplication.processEvents()

        # Raise exception if one occurred in the thread
        if exception[0]:
            raise exception[0]

        return result[0]

    return wrapper


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

        ## 初始化
        self.comboBox_Smode.setCurrentText(conf.base_mode)
        ## conf 初始化部分参数
        # self.comboBox_std_d.setText(conf.std_d)
        conf.conf.set("default", "std_d", self.comboBox_std_d.text())
        with open(conf.ini_path, "w", encoding="utf-8") as conf_ini:
            conf.conf.write(conf_ini)

        conf.reload()  # 重新读取配置文件，以支持热修改

        translator = QTranslator()
        translator.load("src/zh_CN.qm")
        app.installTranslator(translator)
        self.retranslateUi(self)

        if conf.machine in ("HDX", "3D"):
            self.stitch_chip.setEnabled(False)
        pass

    def setupWinShow(self):
        self.setMinimumSize(800, 600)
        self.centralwidget.setStyleSheet('#centralwidget{background-color:#FAFBF3;}')

        self.left_scrollArea.setStyleSheet('#widget_left{background-color:#F5F9E3;}')
        self.left_scrollArea.setFixedWidth(360)
        self.toolBox.setCurrentIndex(0)
        # print(self.widget_right)

        self.widget_right = ChipRegionWidget(self.centralwidget)
        self.widget_right.setObjectName("widget_right")
        self.gridLayout_main.addWidget(self.widget_right, 0, 1, 1, 1)
        self.widget_right.setStyleSheet('#widget_right{background-color:#FF0000;}')

        self.setWindowTitle('BMChiper V4.0')
        self.setWindowIcon(QIcon('./src/bmk_logo.ico'))  # 设置窗体标题图标

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

    @show_wait_dialog
    def open_he_img(self, filename_choose):
        if filename_choose.endswith(".mrxs"):
            self.stitch_chip.setEnabled(True)
            self.new_stitch_channel.setEnabled(True)
        elif filename_choose.endswith(".svs"):
            self.stitch_chip.setEnabled(False)
            self.new_stitch_channel.setEnabled(True)
        elif filename_choose.endswith(".zarray"):
            filename_choose = os.path.split(filename_choose)[0]
            self.stitch_chip.setEnabled(False)
            self.new_stitch_channel.setEnabled(True)
        else:
            self.stitch_chip.setEnabled(False)
            self.new_stitch_channel.setEnabled(True)
        self.widget_right.read_image(filename_choose, mrxs_read_level=conf.mrxs_read_level)

    def open_he_img_cao(self):
        try:
            conf.reload()
            if not isinstance(self.widget_right, ChipRegionWidget):
                self.chip_region_setup_env()
            filename_choose = FileDirBase.open_file(self,'*.tif *.tiff *.mrxs *.svs *.zarray')
            if filename_choose is None:
                return
            print(f"open img \n {filename_choose}")
            self.open_he_img(filename_choose)
        except Exception as e:
            print(traceback.format_exc(), e)
            print("Error:", e)
        # 记忆选择的路径
        base_dir = os.path.split(filename_choose)[0]
        conf.reload()
        conf.conf.set("default", "base_dir", base_dir)
        with open(conf.ini_path, 'w', encoding="utf-8") as f:
            conf.conf.write(f)
        pass

    @show_wait_dialog
    def open_BH1000_img(self, filename_choose):
        from merge_channels_pics_by_libvips import rotate_whole_tif
        compression = conf.BH1000_compression if conf.BH1000_compression else "jpeg"
        rot = int(conf.BH1000_rot) if conf.BH1000_rot else 90

        filename_choose = rotate_whole_tif(filename_choose, save_path=None, compression=compression, rot=rot)

        self.stitch_chip.setEnabled(False)
        self.new_stitch_channel.setEnabled(True)
        self.widget_right.read_image(filename_choose, mrxs_read_level=conf.mrxs_read_level)

    def open_BH1000_img_cao(self):
        try:
            conf.reload()
            if not isinstance(self.widget_right, ChipRegionWidget):
                self.chip_region_setup_env()
            filename_choose = FileDirBase.open_file(self, '*.ome.tif')
            if filename_choose is None:
                return
            print(f"open BH1000 img \n {filename_choose}")
            self.open_BH1000_img(filename_choose)
        except Exception as e:
            print(traceback.format_exc(), e)
            print("Error:", e)
        # 记忆选择的路径
        base_dir = os.path.split(filename_choose)[0]
        conf.reload()
        conf.conf.set("default", "base_dir", base_dir)
        with open(conf.ini_path, 'w', encoding="utf-8") as f:
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
            print("Please choose 4 points first!")
            return
        try:
            corners = np.asarray(self.widget_right.draw_argvs['rect_points'])
            # camera_resolutions = [(2448, 2048), (2048, 2048)]
            # camera_resolution = camera_resolutions[self.comboBox_camera_type.currentIndex()]
            # print("camera_resolution:" + str(camera_resolution))
            st_img = StitchImg(self.widget_right.image_filename, corners * (2**conf.mrxs_read_level), self.widget_right.channel_show)
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
        result = QtWidgets.QMessageBox.question(self, "Exit", "Confirm that you want to exit.",
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

    def change_std_d_cao(self):
        pass
        std_d = self.comboBox_std_d.text()
        conf.conf.set("default", "std_d", std_d)
        with open(conf.ini_path, "w", encoding="utf-8") as conf_ini:
            conf.conf.write(conf_ini)

        conf.reload()  # 重新读取配置文件，以支持热修改

        print("std_d changed:", conf.std_d)

    def new_stitch_channel_cao(self):
        self.new_stitch_channel_process()

    @show_wait_dialog
    def new_stitch_channel_process(self):
        conf.reload()  # 重新读取配置文件，以支持热修改
        if [-1, -1] in self.widget_right.draw_argvs['rect_points']:
            print("Please choose 4 points first!")
            return
        try:
            if self.widget_right.image_filename.endswith('.svs'):
                # 海德星的倍率是2的倍数
                corners = np.asarray(self.widget_right.draw_argvs['rect_points']) * (2 ** conf.mrxs_read_level)
            if self.widget_right.image_filename.endswith('.tif'):
                if self.widget_right.tif_desize is None:
                    corners = np.asarray(self.widget_right.draw_argvs['rect_points'])
                else:
                    corners = np.asarray(self.widget_right.draw_argvs['rect_points']) * self.widget_right.tif_desize
            else:
                corners = np.asarray(self.widget_right.draw_argvs['rect_points']) * (2**conf.mrxs_read_level)
            if conf.base_mode == "S2000-2":
                # S2000-2图像扫描过程，反着扫描，所以点位反转计算一下
                whole_size = np.array(self.widget_right.draw_argvs['zoom_imgs'][-1].size) * (2**conf.mrxs_read_level)
                corners = whole_size - corners

                # S2000-2图像太大，从源头开始缩减一半尺寸用于计算
                corners = (corners * 0.5).astype(corners.dtype)
            # corners = np.asarray(self.widget_right.draw_argvs['rect_points'])
            print(corners)
            # return
            # 改变 stitch_channal
            if self.widget_right.channel_show != 0:
                stitch_channel = self.widget_right.channel_show - 1
                conf.conf.set("match-imgs", "stitch_channal", str(stitch_channel))
                with open(conf.ini_path, "w", encoding="utf-8") as conf_ini:
                    conf.conf.write(conf_ini)

            conf.reload()  # 重新读取配置文件，以支持热修改
            if self.widget_right.image_filename.endswith('.ome.tif'):
                stitch_path = match_imgs_ome.cut_and_stitch(self.widget_right.image_filename, corners.tolist())
            else:
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

    def change_Smode_cao(self):
        Smode = self.comboBox_Smode.currentText()
        conf.conf.set("default", "base_mode", str(Smode))
        with open(conf.ini_path, "w", encoding="utf-8") as conf_ini:
            conf.conf.write(conf_ini)

        conf.reload()  # 重新读取配置文件，以支持热修改
        print("Smode: {}".format(conf.base_mode))
        pass

    def correct_match_result_cao(self):
        if "correct_match_result" in [thread.name for thread in threading.enumerate()]:
            print("threading is exists")
            return
        self.widget_right.mode = 2
        thread = threading.Thread(target=correct_match_result, name="correct_match_result")
        thread.start()

        pass


def correct_match_result():
    conf.reload()  # 重新读取配置文件，以支持热修改
    tmp = window.widget_right.draw_argvs["mode_2_point"].copy()
    while window.widget_right.draw_argvs["mode_2_point"] == tmp:
        time.sleep(0.2)
    print(window.widget_right.draw_argvs["mode_2_point"])
    print(window.widget_right.draw_argvs['zoom_imgs'][-1].size[0], window.widget_right.draw_argvs['zoom_imgs'][-1].size[1])
    wrong_point = window.widget_right.draw_argvs["mode_2_point"]
    whole_size = window.widget_right.draw_argvs['zoom_imgs'][-1].size[:2]
    wrong_point_norm = np.array(wrong_point) / whole_size
    stitch_json_dir = os.path.split(window.widget_right.image_filename)[0]
    stitch_json_path = os.path.join(stitch_json_dir, "stitch_json.json")

    stitch_json = load_json(stitch_json_path)
    if "pyramid_img_path" in stitch_json:  # 如果是ome格式
        match_imgs_ome.correct_img(wrong_point_norm, stitch_json_path)
    else:
        match_imgs.correct_img(wrong_point_norm, stitch_json_path)





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ChipRegionMain()
    window.show()
    sys.exit(app.exec_())

