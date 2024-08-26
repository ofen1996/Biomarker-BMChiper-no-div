import os.path
import sys
import zarr
import openslide
from PyQt5 import QtWidgets
from PyQt5.Qt import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image, ImageQt
import random
import numpy as np
import cv2
import tifffile
import tiffslide
from need.MRXSBase import MRXSBase


def my_warpPerspective(src, M, dsize, **kwargs):
    # 对于尺寸超出限制的图像，采取先缩放，再映射，再放大回原尺寸
    max_len = max(src.shape)
    if max_len > 32768:
        from skimage import transform
        warped_image = transform.warp(src, np.linalg.inv(M), output_shape=dsize[::-1])
        return (warped_image * 255).astype(np.uint8)

        # scale_rate = 20000 / max_len
        # scale_img = cv2.resize(src, (int(src.shape[1] * scale_rate), int(src.shape[0] * scale_rate)))
        # scale_img = cv2.warpPerspective(scale_img, M, scale_img.shape[:2][::-1], **kwargs)
        # return cv2.resize(scale_img, src.shape[:2][::-1])
    else:
        return cv2.warpPerspective(src, M, dsize, **kwargs)
    pass


class ChipRegionWidget(QWidget):
    def __init__(self, parent=None):
        # super
        super(ChipRegionWidget, self).__init__(parent)

        # init参数设置
        self.img_slide = None
        self.channel_show = 0
        self.init_parameters()

        # init工作模式
        # 0：浏览模式，1：选点模式，默认都是0
        self.mode = 0
        self.isPressed = False

        self.setMouseTracking(True)

        self.image_filename = None

        # 对于大的tif图，做下采样处理
        self.tif_desize = None

        # self.corr_spots = CorrSpotsData()

    def init_parameters(self):
        self.draw_argvs = {}
        # 缩放参数
        # 1.0：表示缩略图的1.0倍
        # 8.0: 表示缩略图的8.0倍，即原始图片
        self.draw_argvs['zoom_levels'] = [1.0, 2.0, 4.0, 8.0, 16.0]
        self.draw_argvs['zoom_level_num'] = 5       # 即有N个水平，与zoom_levels对应
        self.draw_argvs['zoom_curr_index'] = 0      # 当前的zoom的下标，即0对应1.0, 1: 2.0
        self.draw_argvs['zoom_curr_level'] = 1.0    # 当前实际的缩放值
        # zoom_imgs为存放的各水平的图片
        self.draw_argvs['zoom_imgs'] = list(map(lambda x: None,range(self.draw_argvs['zoom_level_num'])))
        # 当前展示窗口大小
        self.draw_argvs['draw_win_size'] = [self.width(), self.height()]
        # 当前展示窗口的中心点对应到图片上的位置
        # 加载图片时需要进行初始化，初始到到图片中心
        # 位置均以原始图片尺寸为标准
        # 当用鼠标进行拖动时，中心点不能拖动到图片外去，即最小值为[0,0],最大值为[w,h]
        self.draw_argvs['draw_center_pos'] = [0,0]
        self.draw_argvs['win_center_pos'] = [int(self.width()/2), int(self.height()/2)]
        # 矩形框的4个点坐标，均以原始图片为参考系，坐标点不得超出图片范围
        self.draw_argvs['rect_points'] = [[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
        self.draw_argvs['rect_points_index'] = 0
        # 画十字线的时候，线断的长度，4个方向的长度都一样
        self.draw_argvs['cross_line_len'] = 50
        # 设置实时准心，只有mode1时显示
        self.draw_argvs['curr_mouse'] = [0, 0]
        # 设置一个临时变量，用于储存mode==2状态下的点选坐标
        self.draw_argvs['mode_2_point'] = [-1, -1]
        self.channel_show = 0

        pass

    def win_pos_2_ori_img_pos(self, pos):
        # 计算显示窗口内的偏离，基准点为窗口的中心焦点
        win_center_pos = self.draw_argvs['win_center_pos']
        win_deviate_x = int(pos[0]) - win_center_pos[0]
        win_deviate_y = int(pos[1]) - win_center_pos[1]
        # 获取缩放比例
        zoom_levels = self.draw_argvs['zoom_levels']
        zoom_curr_level = self.draw_argvs['zoom_curr_level']
        # 计算原始图片中的偏离
        if self.img_slide is not None:
            ori_deviate_x = int(win_deviate_x * zoom_levels[0] / zoom_curr_level)
            ori_deviate_y = int(win_deviate_y * zoom_levels[0] / zoom_curr_level)
        else:
            ori_deviate_x = int( win_deviate_x * zoom_levels[-1] / zoom_curr_level )
            ori_deviate_y = int( win_deviate_y * zoom_levels[-1] / zoom_curr_level )
        # 计算原始图片上的坐标
        draw_center_pos = self.draw_argvs['draw_center_pos']
        ori_pos_x = draw_center_pos[0] + ori_deviate_x
        ori_pos_y = draw_center_pos[1] + ori_deviate_y
        # return
        return [ori_pos_x,ori_pos_y]
        pass

    def ori_img_pos_2_win_pos(self, pos):
        # 计算原始图片上的偏离，基准点为图片当前的焦点
        draw_center_pos = self.draw_argvs['draw_center_pos']
        ori_deviate_x = int(pos[0]) - draw_center_pos[0]
        ori_deviate_y = int(pos[1]) - draw_center_pos[1]
        # 获取缩放比例
        zoom_levels = self.draw_argvs['zoom_levels']
        zoom_curr_level = self.draw_argvs['zoom_curr_level']
        # 计算窗口中的偏离值
        win_deviate_x = int( ori_deviate_x * zoom_curr_level / zoom_levels[-1] )
        win_deviate_y = int( ori_deviate_y * zoom_curr_level / zoom_levels[-1] )
        # 计算窗口上的坐标
        win_center_pos = self.draw_argvs['win_center_pos']
        win_pos_x = win_center_pos[0] + win_deviate_x
        win_pos_y = win_center_pos[1] + win_deviate_y
        # return
        return [win_pos_x,win_pos_y]
        pass

    # 注意焦点不得超出原始图片的范围，否则修正回图片内
    def reflush_align_focus(self, win_pos):
        ori_img_pos = self.win_pos_2_ori_img_pos(win_pos)
        # 检测焦点是否在图片外
        if self.img_slide is not None:
            max_x = self.img_slide.level_dimensions[0][0]
            max_y = self.img_slide.level_dimensions[0][1]
        else:
            max_x = self.draw_argvs['zoom_imgs'][-1].size[0]
            max_y = self.draw_argvs['zoom_imgs'][-1].size[1]

        if ori_img_pos[0] < 0:
            ori_img_pos[0] = 0
        if ori_img_pos[0] > max_x:
            ori_img_pos[0] = max_x
        if ori_img_pos[1] < 0:
            ori_img_pos[1] = 0
        if ori_img_pos[1] > max_y:
            ori_img_pos[1] = max_y
        # 基于修正后的原始图片焦点来重新计算窗口焦点
        win_pos_plus = self.ori_img_pos_2_win_pos(ori_img_pos)
        # 更新焦点
        self.draw_argvs['draw_center_pos'] = ori_img_pos
        self.draw_argvs['win_center_pos'] = win_pos_plus
        pass



    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)

        # 绘图的方法就写在这里就好，begin与end之间
        if self.draw_argvs['zoom_imgs'][0] is not None or self.img_slide is not None:
            self.draw_crop_img(qp)
            self.draw_cross_lines(qp)
            self.draw_view_cross_lines(qp)
        else:
            self.drewLines(qp)
        qp.end()
    def draw_one_cross_lines(self, qp, ori_img_pos):
        # 初始化画笔
        # 颜色， 粗细， 线条类型（默认为Qt.SolidLine，即实线）
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        qp.setPen(pen)
        # 获取点在窗口上的位置
        win_pos = self.ori_img_pos_2_win_pos(ori_img_pos)
        # 画下字
        cross_line_len = self.draw_argvs['cross_line_len']
        line_w = [win_pos[0],win_pos[1]-cross_line_len,win_pos[0],win_pos[1]+cross_line_len]
        qp.drawLine(line_w[0],line_w[1],line_w[2],line_w[3])
        line_h = [win_pos[0]-cross_line_len,win_pos[1],win_pos[0]+cross_line_len,win_pos[1]]
        qp.drawLine(line_h[0],line_h[1],line_h[2],line_h[3])
        pass
    def draw_cross_lines(self,qp):
        rect_points = self.draw_argvs['rect_points']
        for i in range(4):
            if rect_points[i][0] < 0:
                continue
            else:
                self.draw_one_cross_lines(qp, rect_points[i])
        pass
    def draw_crop_img(self,qp):
        # 获取窗口大小
        win_w_h = [self.width(), self.height()]
        # 计算选取那张图片
        curr_index = self.draw_argvs['zoom_curr_index']
        # 获取图片中心焦点位置
        ori_center_pos = self.draw_argvs['draw_center_pos']
        if self.img_slide is not None:
            curr_center_pos = [int(ori_center_pos[0] / 1), int(ori_center_pos[1] / 1)]
        else:
            scale_val = self.draw_argvs['zoom_levels'][-(curr_index + 1)]
            curr_center_pos = [int(ori_center_pos[0] / scale_val), int(ori_center_pos[1] / scale_val)]
        # print(f"curr_index: {curr_index}  scale_val: ")
        # 计算图片上要展示的区域
        curr_level = self.draw_argvs['zoom_curr_level']
        curr_index_val = self.draw_argvs['zoom_levels'][curr_index]
        img_w_h = [int(win_w_h[0]*curr_index_val/curr_level), int(win_w_h[1]*curr_index_val/curr_level)]
        win_center_pos = self.draw_argvs['win_center_pos']
        img_deviate = [int(win_center_pos[0]*curr_index_val/curr_level), int(win_center_pos[1]*curr_index_val/curr_level)]
        crop_start_pos = [int(curr_center_pos[0]-img_deviate[0]), int(curr_center_pos[1]-img_deviate[1])]
        crop_rect = [crop_start_pos[0], crop_start_pos[1], crop_start_pos[0]+img_w_h[0], crop_start_pos[1]+img_w_h[1]]
        # 修正图片显示的位置：图片已经进行窗口内部了
        show_rect = [0, 0, win_w_h[0], win_w_h[1]]
        # 绘制当前图片
        if self.img_slide is not None:
            sub_rate = 2 ** (4 - curr_index)  # tifffile 读入的时候，取金字塔图像， 起始点位置如果是负数，会缩小倍数
            region_start_pos = (crop_start_pos[0] if crop_start_pos[0] > 0 else 0,
                                crop_start_pos[1] if crop_start_pos[1] > 0 else 0)
            img_crop = self.img_slide.read_region(region_start_pos, 4 - curr_index, (img_w_h[0], img_w_h[1]))
            # img_crop = img_crop.crop(crop_rect)
        else:
            img = self.draw_argvs['zoom_imgs'][curr_index]
            img_crop = img.crop(crop_rect)
        if self.channel_show != 0:
            img_crop = np.asarray(img_crop)
            img_crop = img_crop[..., self.channel_show - 1]
            img_crop = Image.fromarray(img_crop)
        qt_image = ImageQt.ImageQt(img_crop)
        rect = QRect(show_rect[0],show_rect[1],show_rect[2],show_rect[3])
        qp.drawImage(rect, qt_image)

        pass

    def draw_view_cross_lines(self, qp):
        # 初始化画笔
        # 颜色， 粗细， 线条类型（默认为Qt.SolidLine，即实线）
        # pen = QPen(Qt.red, 2, Qt.SolidLine)
        pen = QPen(Qt.red, 2, Qt.DashDotLine)
        qp.setPen(pen)
        # 获取点在窗口上的位置
        win_pos = self.draw_argvs['curr_mouse']
        # 画下字
        cross_line_len = self.draw_argvs['cross_line_len']*5
        line_w = [win_pos[0],win_pos[1]-cross_line_len,win_pos[0],win_pos[1]+cross_line_len]
        qp.drawLine(line_w[0],line_w[1],line_w[2],line_w[3])
        line_h = [win_pos[0]-cross_line_len,win_pos[1],win_pos[0]+cross_line_len,win_pos[1]]
        qp.drawLine(line_h[0],line_h[1],line_h[2],line_h[3])
        pass



    def drewLines(self, qp):
        pen = QPen(QColor(238, 0, 0), 2, Qt.SolidLine)
        qp.setPen(pen)  # 对画笔进行设置，QColor参数为颜色的rgb值，后面2为线的宽，Qt.SolidLine是线的种类
        # print(self.size())  # 确认下能画画的画布像素范围
        # 随机画几个线段
        for i in range(5):
            # 四个参数分别是横坐标的起始位置，纵坐标的起始位置，横坐标的最终位置，纵坐标的最终位置
            qp.drawLine(random.randint(0, 199), random.randint(0, 199), random.randint(0, 199),
                        random.randint(0, 199))


    def read_image(self, image_filename, mrxs_read_level=0):
        self.img_slide = None
        self.image_filename = image_filename
        # 处理mrxs格式图片
        if image_filename[-4:] == 'MRXS' or image_filename[-4:] == 'mrxs':
            mb = MRXSBase(slide_file=image_filename)
            img = mb.extract_img_by_level(mrxs_read_level)
        elif image_filename[-3:] == 'SVS' or image_filename[-3:] == 'svs':
            slide = openslide.OpenSlide(image_filename)
            img = slide.read_region((0, 0), level=mrxs_read_level, size=slide.level_dimensions[mrxs_read_level])
            img = np.rot90(img, -1)
            img = Image.fromarray(img)
        elif image_filename[-5:] == '-zarr':
            slide = zarr.open(image_filename, mode="r")
            img = Image.fromarray(slide[::2**mrxs_read_level, ::2**mrxs_read_level, ...])
        elif image_filename.endswith(".ome.tif"):
            slide = openslide.open_slide(image_filename)
            img = slide.read_region((0, 0), level=mrxs_read_level, size=slide.level_dimensions[mrxs_read_level])
            self.tif_desize = 2 ** mrxs_read_level

            # self.img_slide = tiffslide.open_slide(image_filename)
            # # 初始焦点设置
            # self.draw_argvs['draw_center_pos'] = [int(self.img_slide.level_dimensions[0][0] / 2),
            #                                       int(self.img_slide.level_dimensions[0][1] / 2)]
            # self.draw_argvs['win_center_pos'] = [int(self.width() / 2), int(self.height() / 2)]
            # return
        # 图片读取
        else:
            # img = tifffile.imread(self.image_filename, level=0)
            with tifffile.TiffFile(self.image_filename) as tif:
                if tif.series[0].shape[0] > 30000 or tif.series[0].shape[1] > 30000:
                    img = tif.asarray(level=0)
                    img = img[::2, ::2, ...]
                    self.tif_desize = 2
                else:
                    img = tif.asarray(level=0)
            img = Image.fromarray(img)
        # 图片缩放
        level_num = self.draw_argvs['zoom_level_num']
        for i in range(level_num):
            if i == level_num - 1:
                self.draw_argvs['zoom_imgs'][i] = img
            else:
                level_scale = self.draw_argvs['zoom_levels'][level_num-1-i]
                newsize = (int(img.size[0] / level_scale), int(img.size[1] / level_scale))
                self.draw_argvs['zoom_imgs'][i] = img.resize(newsize)
        # 初始焦点设置
        self.draw_argvs['draw_center_pos'] = [int(img.size[0]/2), int(img.size[1]/2)]
        self.draw_argvs['win_center_pos'] = [int(self.width() / 2), int(self.height() / 2)]
        pass
    # 重写鼠标点击按下的事件处理函数

    # def read_image(self, image_filename):
    #     self.image_filename = image_filename
    #     # 处理mrxs格式图片
    #     if image_filename[-4:] == 'MRXS' or image_filename[-4:] == 'mrxs':
    #         mb = MRXSBase(slide_file=image_filename)
    #         img_level = 2
    #         img = mb.extract_img_by_level(img_level)
    #         # img = MRXSBase.mrxs_img_to_cv2(img)
    #         tif_file = f"{image_filename[:-5]}_level{img_level}.tif"
    #         cv2.imwrite(tif_file, img)
    #         self.image_filename = tif_file
    #     # 图片读取
    #     img = Image.open(self.image_filename)
    #     # 图片缩放
    #     level_num = self.draw_argvs['zoom_level_num']
    #     for i in range(level_num):
    #         if i == level_num - 1:
    #             self.draw_argvs['zoom_imgs'][i] = img
    #         else:
    #             level_scale = self.draw_argvs['zoom_levels'][level_num-1-i]
    #             newsize = (int(img.size[0] / level_scale), int(img.size[1] / level_scale))
    #             self.draw_argvs['zoom_imgs'][i] = img.resize(newsize)
    #     # 初始焦点设置
    #     self.draw_argvs['draw_center_pos'] = [int(img.size[0]/2), int(img.size[1]/2)]
    #     self.draw_argvs['win_center_pos'] = [int(self.width() / 2), int(self.height() / 2)]
    #     pass
    # 重写鼠标点击按下的事件处理函数
    def mousePressEvent(self, evt):
        print([evt.x(), evt.y()])
        if self.draw_argvs['zoom_imgs'][0] is None and self.img_slide is None:
            return
        # 通过鼠标拖动来移动图片，从而达到浏览图片的目标
        if self.mode == 0 and evt.button() == Qt.LeftButton:
            self.mouse_originX = evt.globalX()
            self.mouse_originY = evt.globalY()
            self.isPressed = True
        # 通过鼠标点击来添加分割点
        if self.mode in [1, 2] and evt.button() == Qt.LeftButton:
            # 获取鼠标点击坐标，并转换为原始图片上的坐标
            win_pos = [evt.x(), evt.y()]
            ori_img_pos = self.win_pos_2_ori_img_pos(win_pos)
            # 检测原始图片上的坐标是否超出图片范围
            if self.img_slide is None:
                if ori_img_pos[0] < 0 or ori_img_pos[0] > self.draw_argvs['zoom_imgs'][-1].size[0]:
                    print('pos out ori img:', ori_img_pos, win_pos)
                    return
                if ori_img_pos[1] < 0 or ori_img_pos[1] > self.draw_argvs['zoom_imgs'][-1].size[1]:
                    print('pos out ori img:', ori_img_pos, win_pos)
                    return
            if self.mode == 1:
                # 保存坐标点
                curr_point_index = self.draw_argvs['rect_points_index']
                self.draw_argvs['rect_points'][curr_point_index][:] = ori_img_pos
            elif self.mode == 2:
                # mode == 2 时，临时点选坐标
                self.draw_argvs['mode_2_point'][:] = ori_img_pos
            # 恢复模式到浏览模式
            self.mode = 0
            # 刷新窗口显示
            self.update()
        pass
    # 重写鼠标移动事件函数
    def mouseMoveEvent(self, evt):
        # print("mouse start move")
        if self.draw_argvs['zoom_imgs'][0] is None and self.img_slide is None:
            return
        # 实时更新鼠标位置
        self.draw_argvs['curr_mouse'] = [evt.x(), evt.y()]
        # print(f"curr_mouse:{self.draw_argvs['curr_mouse']}")
        # self.update()
        # 移动图片
        # time.sleep(0.3)
        if self.mode == 0 and self.isPressed:
            # 计算坐标偏移
            curr_global = [evt.globalX(), evt.globalY()]
            move_x = curr_global[0] - self.mouse_originX
            move_y = curr_global[1] - self.mouse_originY
            self.mouse_originX = curr_global[0]
            self.mouse_originY = curr_global[1]
            # 计算移动后的中心焦点位置
            center_pos = self.draw_argvs['draw_center_pos']
            curr_level = self.draw_argvs['zoom_curr_level']
            max_level = self.draw_argvs['zoom_levels'][-1]
            ori_img_move_x = int(-move_x*max_level/curr_level+center_pos[0])
            ori_img_move_y = int(-move_y*max_level/curr_level+center_pos[1])
            # print(f"ori_img_move_x:{ori_img_move_x}, {ori_img_move_y}")
            # 修正中心焦点位置坐标，其不得超出图片范围
            if self.img_slide is not None:
                max_x = self.img_slide.level_dimensions[0][0]
                max_y = self.img_slide.level_dimensions[0][1]
            else:
                max_x = self.draw_argvs['zoom_imgs'][-1].size[0]
                max_y = self.draw_argvs['zoom_imgs'][-1].size[1]

            if ori_img_move_x < 0:
                ori_img_move_x = 0
            if ori_img_move_x > max_x:
                ori_img_move_x = max_x
            if ori_img_move_y < 0:
                ori_img_move_y = 0
            if ori_img_move_y > max_y:
                ori_img_move_y = max_y
            # 更新中心焦点坐标值
            self.draw_argvs['draw_center_pos'] = [ori_img_move_x, ori_img_move_y]
            # print(f"draw_center_pos:{self.draw_argvs['draw_center_pos']}, {curr_level}, {max_level}")
            # 刷新界面

        # print(f"draw_center_pos:{self.draw_argvs['draw_center_pos']}")
        self.update()

        # print("mouse move")
        pass

    # 重写鼠标释放事件函数
    def mouseReleaseEvent(self, evt):
        print(f"###########mouseReleaseEvent###########")
        if self.draw_argvs['zoom_imgs'][0] is None and self.img_slide is None:
            return
        if self.mode == 0:
            self.isPressed = False
            self.update()
        pass
    # 重写滚轮处理事件函数
    def wheelEvent(self, event):
        # 获取滚轮转过的数值
        angle = event.angleDelta()  # 返回滚轮转过的数值，单位为1/8度.PyQt5.QtCore.QPoint(0, 120)
        angle = angle / 8  # 除以8之后单位为度。PyQt5.QtCore.QPoint(0, 15)   【向前滚是正数，向后滚是负数  用angle.y()取值】
        # 获取鼠标所在的位置
        mouse_pos = event.pos()    # 返回相对于控件的当前鼠标位置.PyQt5.QtCore.QPoint(260, 173)
        # 当处于浏览模式下时，进行放大缩小处理
        if self.mode == 0:
            if angle.y() > 0:
                self.zoom_image(1,[mouse_pos.x(),mouse_pos.y()])
            else:
                self.zoom_image(-1,[mouse_pos.x(),mouse_pos.y()])
            self.isPressed = False
            self.update()
        pass
    # 基于鼠标所在的位置，进行图片的放大与缩小
    # 即以滚轮的动作来进行缩放处理
    def zoom_image(self, zoom_type, win_pos):
        # 获取并保存缩放水平
        curr_level = self.draw_argvs['zoom_curr_level']*(zoom_type*0.4+1.0)
        zoom_level = self.draw_argvs['zoom_levels']
        if curr_level > zoom_level[-1]:
            return
        if curr_level < zoom_level[0]:
            return
        self.draw_argvs['zoom_curr_level'] = curr_level
        # 计算当前图层
        level_num = self.draw_argvs['zoom_level_num']
        level_index = 0
        if curr_level <= (zoom_level[0]+zoom_level[1])/2.0:
            level_index = 0
        elif curr_level > (zoom_level[level_num-2]+zoom_level[level_num-1])/2.0:
            level_index = level_num - 1
        else:
            for i in range(1,level_num-1, 1):
                if curr_level <= (zoom_level[i]+zoom_level[i+1])/2.0 and curr_level > (zoom_level[i]+zoom_level[i-1])/2.0:
                    level_index = i
                    break
        self.draw_argvs['zoom_curr_index'] = level_index
        # 更新展示焦点，移到到鼠标所在的位置
        self.reflush_align_focus(win_pos=win_pos)
        # 刷屏
        self.update()
        pass

    def collect_points_on_ori_img(self, pos):
        # 初始化要采集点的位置
        self.draw_argvs['rect_points_index'] = pos
        self.mode = 1
        print(f"mode:{self.mode}")
        pass

    def min_rect_points(self):
        box_points = np.array(self.draw_argvs['rect_points'])
        min_rect = cv2.minAreaRect(box_points)
        rect_points = cv2.boxPoints(min_rect)
        rect_points = np.int0(rect_points)
        reg_box = list(map(lambda x: [int(x[0]), int(x[1])], rect_points))
        return reg_box
        pass

    def sort_rect_points(self, pts):
        pts = np.array(pts)
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        if leftMost[0, 1] != leftMost[1, 1]:
            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        else:
            leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
        (tl, bl) = leftMost
        if rightMost[0, 1] != rightMost[1, 1]:
            rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        else:
            rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
        (tr, br) = rightMost
        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        new_pts =  np.array([tl, tr, br, bl], dtype="float32")
        rect_points = np.int0(new_pts)
        rect_points = list(map(lambda x: [int(x[0]), int(x[1])], rect_points))
        return rect_points
        pass

    def warp_save_chip_region(self, reg_box):
        # 对四个顶点坐标进行排序
        reg_box = self.sort_rect_points(reg_box)
        # 计算长宽
        img_w = np.sqrt(np.power(reg_box[0][0]-reg_box[1][0], 2.0)+np.power(reg_box[0][1]-reg_box[1][1], 2.0))
        img_h = np.sqrt(np.power(reg_box[0][0]-reg_box[3][0], 2.0)+np.power(reg_box[0][1]-reg_box[3][1], 2.0))
        img_w = round(img_w)
        img_h = round(img_h)

        # 变换矩阵
        desc_box = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        reg_box = self.draw_argvs['rect_points']
        M = self.M_matrix(reg_box, desc_box)
        # 透视变换
        ori_img = self.draw_argvs['zoom_imgs'][-1]
        np_img = np.array(ori_img)
        warped = my_warpPerspective(np_img, M, (img_w, img_h))
        print(f"reg_box={reg_box}, desc_box={desc_box}")

        tif_file = f"{self.image_filename[:-4]}_chip.tif"
        tifffile.imwrite(tif_file, warped, compression="jpeg")
        # cv2.imwrite(tif_file, warped[:,:,[2,1,0]])
        return tif_file


    def M_matrix(self, scr_rect, dst_rect):
        rect = np.array(scr_rect, dtype='float32')
        dst = np.array(dst_rect, dtype='float32')
        # 变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        # return
        return M







if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = WinLeftRightSplitMain(setting=None)
    # submit_values= {'blacks': [0, 4, 4], 'whites': [255, 50, 50], 'gammas': [1.0, 1.69, 1.69], 'binary_cutoff': 100,
    #                 'process_control_str': 'dilate:9,erode:19,dilate:7,erode:20,dilate:27'}
    # window = FromChipSearchMain(setting=submit_values)
    window.show()
    sys.exit(app.exec_())

