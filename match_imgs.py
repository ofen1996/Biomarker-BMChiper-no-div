import os.path

import cv2
import numpy as np
import openslide
import scipy
import skimage.transform
import tifffile
from need.ofen_tool import *
from need.KpDetectByYolo import MyDetector
from need.config import conf
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

detector = MyDetector("./model/best.onnx")
std_edge_size = (2248, 2648)


def binary_pic(pic, median_blur_ksize=11, blockSize=101):
    # floor_light = 10
    # pic = np.where(pic < floor_light, 5, pic)

    if len(pic.shape) == 3:
        B, G, R = cv2.split(pic)
        # 二值化
        B, G, R = map(binary_pic, [B, G, R])
        b_pic = cv2.merge([B, G, R])
        # show_img(b_pic)
        return b_pic

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # show_img(pic)
    pic = clahe.apply(pic)
    # show_img(pic)

    # ret, binary = cv2.threshold(pic, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(pic.mean())
    # show_img(pic)
    brightness = pic.mean()
    # print("brightness:{}".format(brightness))
    if brightness <= 5:
        b_threshold = -1
    elif brightness < 9:
        b_threshold = -2
    elif brightness < 15:
        b_threshold = -3
    elif brightness < 20:
        b_threshold = -4
    elif brightness < 25:
        b_threshold = -5
    else:
        b_threshold = -6
    binary = cv2.adaptiveThreshold(pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, b_threshold)
    # show_img(binary)
    if median_blur_ksize != 0:
        binary = cv2.medianBlur(binary, ksize=median_blur_ksize)
    # binary = cv2.medianBlur(binary, ksize=5)
    # show_img(binary)
    return binary

# 融合2张图像，以a,b的比值作为系数，融合2张图
def merge_two_img(a, b):
    if a.shape != b.shape:
        raise Exception(f"a, b shape is different{a.shape}, {b.shape}")
    # RGB处理方法
    if len(a.shape) == 3:
        result = np.zeros_like(a)
        for channel in range(a.shape[2]):
            result[:, :, channel] = merge_two_img(a[:, :, channel], b[:, :, channel])
        return result

    # gray 图像
    mask = cv2.blur(a, (11, 11)) > cv2.blur(b, (11, 11))
    return np.where(mask, a, b)


def find_div_points_20x(b_img, ignore_pixes=40):
    b_erode_img = cv2.erode(b_img, cv2.getStructuringElement(cv2.MORPH_ERODE, (5, 5)), iterations=1)
    # show_img(b_erode_img)

    b_dilate_img = cv2.dilate(b_erode_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 6)), iterations=3)
    # show_img(b_dilate_img)

    step_lenth = 2
    sum_w = []
    for h in range(b_dilate_img.shape[1]):
        sum_w.append(b_dilate_img[:, h].sum())
    sum_h = []
    for w in range(b_dilate_img.shape[0]):
        sum_h.append(b_dilate_img[w, :].sum())

    sum_w = np.convolve(sum_w, np.ones(step_lenth), 1)
    sum_h = np.convolve(sum_h, np.ones(step_lenth), 1)

    grad_w = abs(np.gradient(sum_w))
    grad_h = abs(np.gradient(sum_h))

    g_w = np.convolve(grad_w, np.ones(9), 1)
    g_h = np.convolve(grad_h, np.ones(9), 1)

    # 归一化
    g_w = g_w / g_w.max()
    g_h = g_h / g_h.max()

    # 附近像素平均值
    sum_w_rate = np.convolve(sum_w, np.ones(20), 1)
    sum_h_rate = np.convolve(sum_h, np.ones(20), 1)

    sum_w_rate = sum_w_rate / (sum_w_rate.max() + 0.00001)
    sum_h_rate = sum_h_rate / (sum_h_rate.max() + 0.00001)

    # 评价指标 梯度大，且sum小
    value_w = g_w * (1 - sum_w_rate)
    value_h = g_h * (1 - sum_h_rate)

    value_w[:ignore_pixes] = 0
    value_w[-ignore_pixes:] = 0
    value_h[:ignore_pixes] = 0
    value_h[-ignore_pixes:] = 0
    # import matplotlib.pyplot as plt
    # plt.plot(value_w)
    # plt.plot(value_h)

    # # # 去掉边界40像素影响，去掉边界40以内的分界线tmp
    # tmp_div_w = list(filter(lambda x: ignore_pixes < x < b_img.shape[1] - ignore_pixes,
    #                         np.where(value_w > value_w.max() * 0.5)[0]))
    # tmp_div_h = list(filter(lambda x: ignore_pixes < x < b_img.shape[0] - ignore_pixes,
    #                         np.where(value_h > value_h.max() * 0.5)[0]))

    return np.argmax(value_w), np.argmax(value_h)


def cut_fov_img(mrxs_path, save_path, camera_resolution=(2448, 2048)):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    slide = openslide.OpenSlide(mrxs_path)

    # 非零起始点位置
    BOUND_X, BOUND_Y = map(int, [slide.properties['openslide.bounds-x'], slide.properties['openslide.bounds-y']])
    # 视场的长宽像素值
    BOUND_WIDTH, BOUND_HEIGHT = map(
        int, [slide.properties['openslide.bounds-width'], slide.properties['openslide.bounds-height']])
    # 视场数目
    FOV_COUNT = int(slide.properties['mirax.NONHIERLAYER_0_SECTION.SCANNED_FOV_COUNT'])
    # 相机固定分辨率
    CAMERA_RESOLUTION = np.asarray(camera_resolution)
    # CAMERA_RESOLUTION = np.asarray((2048, 2048))
    # 视场数目，长，宽。
    FOV_SHAPE = ((BOUND_WIDTH, BOUND_HEIGHT) // (CAMERA_RESOLUTION + 0.00000001) + 1).astype(int)
    # 验证计算的视场长宽与视场数目是否对应，如果不对应，说明计算有误，抛错停止
    if FOV_COUNT % (FOV_SHAPE[0] * FOV_SHAPE[1]) != 0:
        print("FOV_SHAPE: {}".format(FOV_SHAPE))
        print("FOV_COUNT: {}".format(FOV_COUNT))
        if conf.if_raise_exception:
            raise Exception("Fov is wrong!")
    # 计算得到每一个视场的像素值尺寸
    FOV_PIXES = ((BOUND_WIDTH, BOUND_HEIGHT) / FOV_SHAPE).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        if conf.base_mode == "S2000-2":
            # S2000-2图像太大，从源头开始缩减一半尺寸用于计算
            FOV_PIXES = np.array(FOV_PIXES) // 2
        return FOV_SHAPE[::-1], FOV_PIXES[::-1]

    # 先裁剪全图
    if conf.base_mode == "S2000-2":
        # S2000-2图像太大，从源头开始缩减一半尺寸用于计算
        FOV_PIXES = np.array(FOV_PIXES) // 2
        BOUND_WIDTH //= 2
        BOUND_HEIGHT //= 2
        whole_img = np.asarray(slide.read_region((BOUND_X, BOUND_Y), 1, (BOUND_WIDTH, BOUND_HEIGHT)))[..., :3]
        whole_img = np.rot90(whole_img, k=2)

    else:
        whole_img = np.asarray(slide.read_region((BOUND_X, BOUND_Y), 0, (BOUND_WIDTH, BOUND_HEIGHT)))[..., :3]
    # whole_img = binary_pic(whole_img)
    # FOV_PIXES[1] = FOV_PIXES[1]//2
    # FOV_PIXES[0] = FOV_PIXES[0]//2
    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            # select_part = np.asarray((wi, hi))
            t1 = time.time()
            # im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            tmp_h_start, tmp_w_start = hi * FOV_PIXES[1], wi * FOV_PIXES[0]
            im = whole_img[tmp_h_start:tmp_h_start+FOV_PIXES[1], tmp_w_start:tmp_w_start+FOV_PIXES[0]]
            ### 以下为免疫荧光专用调试代码，需要注释
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            # im[..., conf.stitch_channal] = clahe.apply(im[..., conf.stitch_channal])
            # im[..., conf.stitch_channal] = gray_enhance(im[..., conf.stitch_channal], black=93, white=119, gamma=1.37)

            ####
            t2 = time.time()
            # show_img(im)
            save_name = "ori_{}_{}.tif".format(hi, wi)
            tifffile.imwrite(os.path.join(save_path, save_name), im)
            t3 = time.time()
    return FOV_SHAPE[::-1], FOV_PIXES[::-1]


def cut_fov_img_tif(tif_path, save_path, camera_resolution=(1216, 1012)):
    # 通用tif
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    FOV_PIXES = camera_resolution

    whole_img = tifffile.imread(tif_path)
    # whole_img = cv2.resize(whole_img, (23040, 22528))

    FOV_SHAPE = (np.asarray(whole_img.shape[:2][::-1]) / FOV_PIXES).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        return FOV_SHAPE[::-1], FOV_PIXES[::-1]

    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            # select_part = np.asarray((wi, hi))
            t1 = time.time()
            # im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            tmp_h_start, tmp_w_start = hi * FOV_PIXES[1], wi * FOV_PIXES[0]
            im = whole_img[tmp_h_start:tmp_h_start+FOV_PIXES[1], tmp_w_start:tmp_w_start+FOV_PIXES[0]]
            ### 以下为免疫荧光专用调试代码，需要注释
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            # im[..., conf.stitch_channal] = clahe.apply(im[..., conf.stitch_channal])
            ####
            t2 = time.time()
            # show_img(im)
            save_name = "ori_{}_{}.tif".format(hi, wi)
            tifffile.imwrite(os.path.join(save_path, save_name), im, compression='LZW')
            # tifffile.imwrite(os.path.join(save_path, save_name), im)
            t3 = time.time()
    return FOV_SHAPE[::-1], FOV_PIXES[::-1]


def cut_fov_img_tif_HDX(tif_path, save_path, camera_resolution=(1216, 1012)):
    # 海德星tif
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    FOV_PIXES = camera_resolution[::-1]  # 因为海德星的图像旋转了90°，所以高和宽换一下

    whole_img = tifffile.imread(tif_path)

    FOV_SHAPE = (np.asarray(whole_img.shape[:2][::-1]) / FOV_PIXES).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        return FOV_SHAPE[::-1], FOV_PIXES[::-1]

    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            # select_part = np.asarray((wi, hi))
            t1 = time.time()
            # im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            tmp_h_start, tmp_w_start = hi * FOV_PIXES[1], wi * FOV_PIXES[0]
            im = whole_img[tmp_h_start:tmp_h_start+FOV_PIXES[1], tmp_w_start:tmp_w_start+FOV_PIXES[0]]
            ### 以下为免疫荧光专用调试代码，需要注释
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            # im[..., conf.stitch_channal] = clahe.apply(im[..., conf.stitch_channal])
            ####
            t2 = time.time()
            # show_img(im)
            save_name = "ori_{}_{}.tif".format(hi, wi)
            tifffile.imwrite(os.path.join(save_path, save_name), im, compression='LZW')
            # tifffile.imwrite(os.path.join(save_path, save_name), im)
            t3 = time.time()
    return FOV_SHAPE[::-1], FOV_PIXES[::-1]


def cut_fov_img_haidexing(svs_path, save_path, camera_resolution=(2448, 2048)):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # FOV_PIXES = camera_resolution[::-1]  # 因为海德星的图像旋转了90°，所以高和宽换一下
    FOV_PIXES = camera_resolution

    slide = openslide.OpenSlide(svs_path)

    FOV_SHAPE = (np.asarray(slide.level_dimensions[0][::-1]) / FOV_PIXES).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        return FOV_SHAPE[::-1], FOV_PIXES[::-1]

    whole_img = np.asarray(slide.read_region((0, 0), level=0, size=slide.level_dimensions[0]))[..., :3]
    whole_img = np.rot90(whole_img, -1)  # 因为海德星的图像旋转了90°

    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            # select_part = np.asarray((wi, hi))
            t1 = time.time()
            # im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            tmp_h_start, tmp_w_start = hi * FOV_PIXES[1], wi * FOV_PIXES[0]
            im = whole_img[tmp_h_start:tmp_h_start+FOV_PIXES[1], tmp_w_start:tmp_w_start+FOV_PIXES[0]]
            ### 以下为免疫荧光专用调试代码，需要注释
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            # im[..., conf.stitch_channal] = clahe.apply(im[..., conf.stitch_channal])
            ####
            t2 = time.time()
            # show_img(im)
            save_name = "ori_{}_{}.tif".format(hi, wi)
            tifffile.imwrite(os.path.join(save_path, save_name), im, compression='LZW')
            # tifffile.imwrite(os.path.join(save_path, save_name), im)
            t3 = time.time()
    return FOV_SHAPE[::-1], FOV_PIXES[::-1]


def cut_fov_img_S2000(mrxs_path, save_path, camera_resolution=(2448, 2048)):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    slide = openslide.OpenSlide(mrxs_path)

    # 非零起始点位置
    BOUND_X, BOUND_Y = map(int, [slide.properties['openslide.bounds-x'], slide.properties['openslide.bounds-y']])
    # 视场的长宽像素值
    BOUND_WIDTH, BOUND_HEIGHT = map(
        int, [slide.properties['openslide.bounds-width'], slide.properties['openslide.bounds-height']])
    # 视场数目
    FOV_COUNT = int(slide.properties['mirax.NONHIERLAYER_0_SECTION.SCANNED_FOV_COUNT'])
    # 相机固定分辨率
    CAMERA_RESOLUTION = np.asarray(camera_resolution)
    # CAMERA_RESOLUTION = np.asarray((2048, 2048))
    # 视场数目，长，宽。
    FOV_SHAPE = ((BOUND_WIDTH, BOUND_HEIGHT) // (CAMERA_RESOLUTION + 0.00000001) + 1).astype(int)
    # 验证计算的视场长宽与视场数目是否对应，如果不对应，说明计算有误，抛错停止
    if FOV_COUNT % (FOV_SHAPE[0] * FOV_SHAPE[1]) != 0:
        print("FOV_SHAPE: {}".format(FOV_SHAPE))
        print("FOV_COUNT: {}".format(FOV_COUNT))
    # 计算得到每一个视场的像素值尺寸
    FOV_PIXES = ((BOUND_WIDTH, BOUND_HEIGHT) / FOV_SHAPE).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        return FOV_SHAPE[::-1], FOV_PIXES[::-1]

    # 先裁剪全图
    # whole_img = np.array(slide.read_region((BOUND_X, BOUND_Y), 0, (BOUND_WIDTH, BOUND_HEIGHT)))
    # whole_img = whole_img[..., :3]
    # stable_binary_pic(whole_img)

    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            # select_part = np.asarray((wi, hi))
            t1 = time.time()
            # im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            tmp_h_start, tmp_w_start = BOUND_Y + hi * FOV_PIXES[1], BOUND_X + wi * FOV_PIXES[0]
            # im = whole_img[tmp_h_start:tmp_h_start+FOV_PIXES[1], tmp_w_start:tmp_w_start+FOV_PIXES[0]]
            im = np.array(slide.read_region((tmp_w_start, tmp_h_start), 0, (FOV_PIXES[0], FOV_PIXES[1])))[..., :3]
            # binary_pic(im)
            t2 = time.time()
            # show_img(im)
            if conf.base_mode == "S2000-2":
                save_name = "ori_{}_{}.tif".format(FOV_SHAPE[1]-hi-1, FOV_SHAPE[0]-wi-1)
                tifffile.imwrite(os.path.join(save_path, save_name), np.rot90(im, k=2), compression="LZW")
            else:
                save_name = "ori_{}_{}.tif".format(hi, wi)
                tifffile.imwrite(os.path.join(save_path, save_name), im, compression="LZW")
            t3 = time.time()
    return FOV_SHAPE[::-1], FOV_PIXES[::-1]


class StdCircles:
    def __init__(self, ori_size, shape, d, r, M=None):
        self.circles_img, self.circles_mask, self.circle_centers = self.std_circles(ori_size, shape, d, r,
                                                                                    tile_limit=(conf.base_size_x,
                                                                                                conf.base_size_y))
        self.reg_box = [self.circle_centers[0, 0].tolist(),
                        self.circle_centers[(conf.barcode_size_x + 1) * conf.base_size_x,
                                            (conf.barcode_size_y + 1) * conf.base_size_y].tolist()]
        self.reg_box[0][0] += round(d//2)
        self.reg_box[1][0] += round(d//2)

        if M is not None:  # 对标准板做逆变换，使其在缝合过程避免变换丢失数据，最后再变换回正确图案
            M_inv = np.linalg.inv(M)
            # self.circles_img = cv2.warpPerspective(self.circles_img, M_inv, self.circles_img.shape[:2][::-1])
            self.circles_img = my_warpPerspective(self.circles_img, M_inv, self.circles_img.shape[:2][::-1])
            self.circles_mask = my_warpPerspective(self.circles_mask, M_inv, self.circles_mask.shape[:2][::-1])
            # 先调换一下维度顺序，在转换，再换回来
            self.circle_centers = np.round(cv2.perspectiveTransform(
                np.array(self.circle_centers, dtype=np.float32).reshape(-1, 1, 2), M_inv).reshape(
                *self.circle_centers.shape)).astype(int)
        # tmp = np.where(self.circles_mask > 0, 255, 0).astype(np.uint8)
        # edge_rect_dilate_a = int(conf.conf.get("std-template", "edge_rect_dilate_a"))
        # edge_rect_dilate_b = int(conf.conf.get("std-template", "edge_rect_dilate_b"))
        # tmp_a = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_DILATE, (edge_rect_dilate_a, edge_rect_dilate_a)))
        # tmp_b = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_RECT, (edge_rect_dilate_b, edge_rect_dilate_b)))
        # self.edge_rect = tmp_b - tmp_a
        # cv2.imwrite("test_circles.tif", self.circles_img)
        cv2.imwrite("test_circles_mask.tif", self.circles_mask)
        # cv2.imwrite("edge_rect.tif", self.edge_rect)

        # 计算首个分割点位置
        temp_delta_y = (self.circle_centers[conf.barcode_size_x, conf.barcode_size_y + 2][1] -
                        self.circle_centers[conf.barcode_size_x, conf.barcode_size_y][1]) // 2
        temp_delta_x = (self.circle_centers[conf.barcode_size_x + 2, conf.barcode_size_y - 1][0] -
                        self.circle_centers[conf.barcode_size_x, conf.barcode_size_y][0]) // 2
        self.std_first_div_point = (self.circle_centers[conf.barcode_size_x, conf.barcode_size_y] + (temp_delta_x, temp_delta_y))[::-1]
        # self.std_first_div_point = (self.circle_centers[30, 35] + (26, 30))[::-1]

        # 标准单个区域模板
        # 先计算首个分割点的对角点位置
        tmp_corner_div_point = (self.circle_centers[conf.barcode_size_x*2+1, conf.barcode_size_y*2+1] + (temp_delta_x, temp_delta_y))[::-1]
        # print(tmp_corner_div_point - self.std_first_div_point)
        self.std_mask_size = (tmp_corner_div_point - self.std_first_div_point).tolist()
        # self.std_mask_size = [1081, 1076]
        # conf.std_distance = self.std_mask_size[0]
        self.std_mask = self.circles_img[self.std_first_div_point[0]:self.std_first_div_point[0] + self.std_mask_size[0],
                                         self.std_first_div_point[1]:self.std_first_div_point[1] + self.std_mask_size[1]]
        self.std_mask_centers = self.circle_centers[conf.barcode_size_x + 2: conf.barcode_size_x + 2 + conf.barcode_size_x,
                                                    conf.barcode_size_y + 2: conf.barcode_size_y + 2 + conf.barcode_size_y] - self.std_first_div_point[::-1]


    @staticmethod
    def std_circles(ori_size, shape, d, r, tile_limit=(46, 46)):
        # std_circles((1200, 1200), (30, 35), 35, 10)
        size = (ori_size[0] + 30, ori_size[1] + 30)
        circles_img = np.zeros(size, dtype=np.uint8)
        circles_mask = np.zeros(size, dtype=np.uint8)
        delta_x = d
        delta_y = d * 0.5 * 3 ** 0.5
        total_circles_shape = (int(size[0] / delta_y) + 1, int(size[1] / delta_x) + 1)[::-1]
        # first_circle_loc = (np.asarray(size) - (np.asarray(shape) - 1) * np.asarray((delta_x, delta_y))) // 2
        # first_circle_loc = np.asarray((delta_x, delta_y), dtype=int)
        first_circle_loc = np.asarray((conf.std_edge_size[1], conf.std_edge_size[0]), dtype=int)

        circle_centers = np.zeros((*total_circles_shape, 2), dtype=int)
        for y in range(total_circles_shape[1]):
            for x in range(total_circles_shape[0]):

                rate = 0.8
                if y % (shape[1] + 1) in (1, shape[1]) or x % (shape[0] + 1) in (1, shape[0]):
                    rate = 1  # 对边缘最近的一行的权重提高，这样在计算时候可以让匹配尽量贴近边缘，避免边缘不完整图像匹配出错
                first_loc = first_circle_loc
                if y % 2 == 0:
                    first_loc = first_circle_loc - (d / 2, 0)
                tmp_center = (first_loc + (x * delta_x, y * delta_y)).astype(int)
                # circle_centers.append(tmp_center.astype(int))
                circle_centers[x, y, :] = tmp_center
                if y >= (shape[1] + 1) * tile_limit[1] or x >= (shape[0] + 1) * tile_limit[0]:
                    # 超过芯片边界的不要
                    continue
                if y % (shape[1] + 1) == 0 or x % (shape[0] + 1) == 0:
                    # 不同块的分割线
                    continue
                if y <= 5 and x <= 10:
                    # 左上角缺块
                    continue
                if y <=11 and (13 - y)//2 + x >= (shape[0] + 1) * tile_limit[0]:
                    # 右上角缺口
                    continue
                if y >= (shape[1] + 1) * tile_limit[1] - 5 and x <= 5:
                    # 左下角缺口
                    continue
                circles_mask = cv2.circle(circles_mask, tuple(tmp_center), r+r-1, int(255 * rate), -1)
                # circles_mask = cv2.circle(circles_mask, tuple(tmp_center), r-3, int(255 * rate * 0.3), -1)
                circles_img = cv2.circle(circles_img, tuple(tmp_center), r, 255, 1)
        # print(circle_centers)
        # for center in circle_centers.reshape(total_circles_shape[0] * total_circles_shape[1], -1):
        #     circles_mask = cv2.circle(circles_mask, tuple(center), r, 255, -1)
        #     circles_img = cv2.circle(circles_img, tuple(center), r, 255, 1)
        # show_img(circles_img)
        # show_img(circles_mask)
        circles_mask = cv2.erode(circles_mask, cv2.getStructuringElement(cv2.MORPH_ERODE, (r-1, r-1)))
        circles_mask = cv2.bitwise_not(circles_mask)
        return [circles_img[:ori_size[0], :ori_size[1]],
                circles_mask[:ori_size[0], :ori_size[1]],
                circle_centers]

def two_point_dist(pt1, pt2):
    pow1 = np.power(pt2[0] - pt1[0], 2.0)
    pow2 = np.power(pt2[1] - pt1[1], 2.0)
    # print(pt1,pt2,pow1,pow2)
    return np.sqrt(pow1 + pow2)


def M_matrix(scr_rect, dst_rect):
    rect = np.array(scr_rect, dtype='float32')
    dst = np.array(dst_rect, dtype='float32')
    # 变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # return
    return M


def calculate_M(reg_box):
    reg_box = np.asarray(reg_box)
    mrxs_rect = reg_box - reg_box[0]
    # mrxs_rect = copy.deepcopy(tmp_reg_box)
    img_wh = [0, 0]
    img_wh[0] = round(two_point_dist(mrxs_rect[0], mrxs_rect[1]))
    img_wh[1] = round(two_point_dist(mrxs_rect[0], mrxs_rect[3]))
    nimg_rect = [[0, 0], [img_wh[0] - 1, 0], [img_wh[0] - 1, img_wh[1] - 1], [0, img_wh[1] - 1]]

    # 计算M变换矩阵
    M = M_matrix(mrxs_rect, nimg_rect)
    return M, nimg_rect


def gen_index_by_reg_box(reg_box, img_shape):
    reg_box = np.asarray(reg_box)
    x_start = min(reg_box[0][0], reg_box[3][0]) // img_shape[1]
    y_start = min(reg_box[0][1], reg_box[1][1]) // img_shape[0]

    x_end = (max(reg_box[2][0], reg_box[1][0])-1) // img_shape[1]
    y_end = (max(reg_box[2][1], reg_box[3][1])-1) // img_shape[0]

    all_index_y_x = []
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            all_index_y_x.append([y, x])
    return [[int(x_start), int(y_start)], [int(x_end), int(y_end)]], all_index_y_x


def rotate_image(image, angle):
    """
    Rotate the image by a given angle and return the rotation matrix and the rotated image.

    Parameters:
    image (np.array): Input image (numpy array).
    angle (float): Rotation angle in degrees, counterclockwise.

    Returns:
    tuple: A tuple containing:
        - np.array: 3x3 rotation matrix
        - np.array: Rotated image
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算图像中心点
    center = (width / 2, height / 2)

    # 使用cv2.getRotationMatrix2D获得2x3的旋转矩阵
    rotation_matrix_2x3 = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 将2x3矩阵扩展成3x3矩阵
    rotation_matrix_3x3 = np.vstack([rotation_matrix_2x3, [0, 0, 1]])

    # 使用旋转矩阵来旋转图像
    rotated_image = cv2.warpAffine(image, rotation_matrix_2x3, (width, height))
    return rotation_matrix_3x3, rotated_image


def find_best_rotate(img):
    # 检查图像是否正确读取
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    angles = []
    grad_values = []
    rot_matrixs = []
    for angle in np.arange(-3, 3, 0.05):
        # 旋转图像
        angle = np.round(angle, 2)  # 旋转角度
        rot_matrix, rot_img = rotate_image(img, angle)

        # 计算每行求和后的二阶梯度之和
        row_sum = np.sum(rot_img[200:-200, 200:-200], axis=1)
        value = sum(np.abs(np.gradient(row_sum, 2)))

        angles.append(angle)
        grad_values.append(value)
        rot_matrixs.append(rot_matrix)

        # print(f"Angle:{angle},Grad value:{value}")

        # show_img(rot_img)
        # 打印旋转矩阵
        # print("Rotation Matrix:\n", rot_matrix)

    # plt.plot(angles, grad_values, marker='o')
    best_angle = angles[np.argmax(grad_values)]
    best_rot_matrix = rot_matrixs[np.argmax(grad_values)]

    # print(f"Best angle:{best_angle}")

    # rotated_image = cv2.warpPerspective(img, best_rot_matrix, img.shape[:2][::-1])
    # show_img(rotated_image)

    return best_angle, best_rot_matrix


def caculate_angle_M(pics_dir, stitch_json):
    x_y_range = stitch_json["x_y_range"]

    all_angle = []
    all_M = []
    for i in range(5):
        # 随机选择序号，避开边缘
        random_x = random.randint(x_y_range[0][0] + 1, x_y_range[1][0] - 1)
        random_y = random.randint(x_y_range[0][1] + 1, x_y_range[1][1] - 1)
        img = cv2.imread(os.path.join(pics_dir, f"ori_{random_y}_{random_x}.tif"))
        best_angle, best_rot_matrix = find_best_rotate(img)
        all_angle.append(best_angle)
        all_M.append(best_rot_matrix)
        print(f"ori_{random_y}_{random_x}.tif -> best_angle:{best_angle}")
    all_angle = np.array(all_angle)
    median_angle = np.median(all_angle)
    median_index = np.where(all_angle == median_angle)[0]  # 取中位数索引
    median_M = all_M[median_index[0]]
    return median_angle, median_M


def new_stitch(pics_dir, reg_box, pic_shape=(2048, 2448), save_dir=None):
    if save_dir is None:
        save_dir = pic_dir + '-new_stitch'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if os.path.exists(os.path.join(save_dir, "stitch_json.json")):
        print("stitch_json is exists, Start stitch img by stitch_json.json")
        draw_img_by_json(pics_dir, os.path.join(save_dir, "stitch_json.json"), save_dir)
        return os.path.join(save_dir, "img_dist_with_board.tif")
    reg_box = np.asarray(reg_box)
    stitch_json = {"reg_box": reg_box.tolist()}
    # M, rect_points = calculate_M(reg_box)
    # if conf.machine == "HDX":
    #     M = np.eye(3)  # 不做变形
    # #######博奥###############
    # M = [
    #     [
    #         0.99998629,
    #         -0.00523596,
    #         5.03639486
    #     ],
    #     [
    #         0.00523596,
    #         0.99998629,
    #         -3.75673452
    #     ],
    #     [
    #         0,
    #         0,
    #         1.0
    #     ]
    # ]
    # M = np.array(M)
    # #####################
    stitch_json["pic_shape"] = np.asarray(pic_shape, dtype=int).tolist()
    # stitch_json["M"] = M.tolist()
    stitch_json["pics_dir"] = pics_dir

    x_y_range, all_index_y_x = gen_index_by_reg_box(reg_box, pic_shape)
    stitch_json["x_y_range"], stitch_json["all_index_y_x"] = x_y_range, all_index_y_x

    # 估算图像偏转角度
    if conf.machine == "3D":
        M, rect_points = calculate_M(reg_box)
    elif conf.machine == "None":
        M = np.eye(3)  # 不做变形
    else:
        angle, M = caculate_angle_M(pics_dir, stitch_json)
        print("final angle: {}".format(angle))
        stitch_json["angle"] = float(angle)

    stitch_json["M"] = M.tolist()

    # 17.565
    # overlap_rate = 0.014297385620915032
    # std_d = rect_points[2][0] / ((31*46)-1)
    # std_d = (rect_points[2][0] - pic_shape[1]*overlap_rate*(46-1)) / ((31*46)-1)
    # std_d = 14.83
    # std_d = 14.836363636363636
    std_d = calculate_std_d(pics_dir, stitch_json)

    if conf.std_d == "auto":
        pass
    else:
        print("Use given std_d!")
        std_d = float(conf.std_d)
    # std_d = 14.836363636363636
    # std_d = 29.237410071942445
    std_r = int(std_d * 0.4)
    print("std_d :{}".format(std_d))
    # 重新设定whole_img_size
    whole_img_size = (int(conf.std_edge_size[1] * 2 + std_d * 0.5 * np.sqrt(3) * (conf.barcode_size_y + 1) * conf.base_size_y) + 1,
                      int(conf.std_edge_size[0] * 2 + std_d * (conf.barcode_size_x + 1) * conf.base_size_x) + 1
                      )
    conf.conf.set("match-imgs", "whole_img_size", str(whole_img_size))
    with open(conf.ini_path, "w") as conf_ini:
        conf.conf.write(conf_ini)
    conf.whole_img_size = whole_img_size

    std_circle = StdCircles(conf.whole_img_size, (conf.barcode_size_x, conf.barcode_size_y), std_d, std_r, M=M)
    stitch_json["std_reg_box"] = std_circle.reg_box
    stitch_json["std_circles_d"] = std_d
    stitch_json["std_circles_center_path"] = os.path.join(save_dir, "circle_centers.npy")

    np.save(stitch_json["std_circles_center_path"], std_circle.circle_centers)

    print("---end draw std circles")
    # stitch_img = cv2.cvtColor(std_circle.circles_mask, cv2.COLOR_GRAY2BGR)
    stitch_img = np.zeros((*conf.whole_img_size, 3), dtype=np.uint8)

    # whole_start_loc = reg_box[0]
    # whole_end_loc = reg_box[2]
    print("---end draw std stitch_img")
    print("start match img---")

    temp_circles_mask = std_circle.circles_mask.copy()

    # 根据全图坐标和 reg_box位置，预测该关键点所在的区域,通过线性拟合后预测位置
    model = LinearRegression()
    X = np.array(reg_box)
    Y = np.asarray([[0, 0], [conf.base_size_x, 0], [conf.base_size_x, conf.base_size_y], [0, conf.base_size_y]])

    if not os.path.exists(os.path.join(save_dir, "stitch_json_ori.json")):
        print("Start match stitch_json_ori.json")

        for index_y_x in all_index_y_x:

            index_y, index_x = index_y_x
            # if not os.path.exists(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x))):
            #     print("ori_{}_{}.tif is not exist".format(index_y, index_x))
            #     continue
            img = tifffile.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))[..., :3]
            # img = cv2.warpPerspective(img_ori, M, img_ori.shape[:2][::-1], borderMode=cv2.BORDER_REFLECT_101)

            # img = img_ori.copy()
            # out_img, centers = detector.detect(img, 0.4)
            img_merge = cv2.cvtColor(img[..., conf.stitch_channal], cv2.COLOR_GRAY2BGR)
            # img_merge = cv2.cvtColor(cv2.equalizeHist(img[..., conf.stitch_channal]), cv2.COLOR_GRAY2BGR)
            # show_img(cv2.equalizeHist(img[..., conf.stitch_channal]))
            # # yolo预测交汇点
            out_img, centers = detector.detect(img_merge, 0.4)  # 荧光解码文件的预测方法
            # out_img, centers = detector.detect(cv2.bitwise_not(binary_pic(img_merge)), 0.4)  # 荧光解码文件的预测方法
            # show_img(out_img)
            if not centers:
                print("Warnning: ori_{}_{}.tif has 0 Key Point, Skip it.".format(index_y, index_x))
                continue
            max_match_kp = centers[0][:2]

            # 梯度预测交汇点
            # max_match_kp = find_div_points_20x(img_merge)
            max_match_kp = np.asarray(max_match_kp) + (23, 18)
            # print(max_match_kp)
            # show_img(img_merge)
            # 计算max_match_kp在整个图像中的空间坐标
            max_match_kp_loc = [index_x * img.shape[1] + max_match_kp[0], index_y * img.shape[0] + max_match_kp[1]]

            # 根据全图坐标和 reg_box位置，预测该关键点所在的区域,通过线性拟合后预测位置
            model.fit(X, Y)
            predect_tile = np.round(model.predict([max_match_kp_loc])).astype(int)
            tile_index_x, tile_index_y = predect_tile[0]
            if tile_index_x < 0 or tile_index_y < 0:
                print("Warnning: ori_{}_{}.tif predect_tile < 0 :{}, Skip it.".format(
                    index_y, index_x, predect_tile))
                continue
            # # 计算max_match_kp相对芯片左上角的坐标
            # max_match_kp_rel = np.asarray(max_match_kp_loc) - reg_box[0]
            #
            # # 下面估算预测的关键点在std_mask底板中的位置
            # distance = (np.asarray(reg_box[2]) - reg_box[0]) // np.array([conf.base_size_x, conf.base_size_y])
            # tile_index_x, tile_index_y = np.round(max_match_kp_rel / distance).astype(int)
            kp_loc = std_circle.circle_centers[tile_index_x * (conf.barcode_size_x+1) + 1,
                                               tile_index_y * (conf.barcode_size_y+1) + 1]  # 找到对应块的第一个圆心坐标(x,y)，近似对应kp位置
            # show_img(std_circle.circles_mask[kp_loc[1]:kp_loc[1]+400, kp_loc[0]:kp_loc[0]+400])

            # 从mask里面截取模板template，然后精准匹配
            template_start_loc = np.asarray(kp_loc) - max_match_kp


            # ### 调试代码####
            # ### 调试代码####
            # tmp_circle = std_circle.circles_img[template_start_loc[1]:template_start_loc[1]+img_ori.shape[0],
            #                                     template_start_loc[0]:template_start_loc[0]+img_ori.shape[1]]
            # show_img(img_bin + cv2.cvtColor(tmp_circle, cv2.COLOR_GRAY2BGR))

            ##############
            match_range = int(conf.conf.get("match-imgs", "match_range"))
            template = std_circle.circles_mask[template_start_loc[1]-match_range:template_start_loc[1]+img.shape[0]+match_range,
                                               template_start_loc[0]-match_range:template_start_loc[0]+img.shape[1]+match_range]
            if template.shape != tuple(np.asarray(img.shape[:2]) + match_range * 2):
                print("Different shape! template shape:{}, img shape shape:{}".format(template.shape, img.shape[:2]))
                continue
            # match_result = cv2.matchTemplate(template, cv2.bitwise_not(img_merge[..., conf.stitch_channal]), cv2.TM_SQDIFF)
            # match_shift = cv2.minMaxLoc(match_result)[2] - np.asarray([match_range, match_range])
            match_result = cv2.matchTemplate(template, img_merge[..., conf.stitch_channal], cv2.TM_CCOEFF)
            match_shift = cv2.minMaxLoc(match_result)[3] - np.asarray([match_range, match_range])
            real_loc = template_start_loc + match_shift

            # ### 调试代码####
            # tmp_circle_shift = std_circle.circles_img[real_loc[1]:real_loc[1]+img_merge.shape[0],
            #                                           real_loc[0]:real_loc[0]+img_merge.shape[1]]
            # show_img(img_merge + cv2.cvtColor(tmp_circle_shift, cv2.COLOR_GRAY2BGR))
            # tifffile.imwrite(os.path.join(save_dir, "img_merge.tif"), img_merge)
            # tifffile.imwrite(os.path.join(save_dir, "template.tif"), template)
            ##############

            print("Match ori_{}_{}.tif, div_point:{}, shift:{}, match rate:{}"
                  "".format(index_y, index_x, max_match_kp, match_shift, cv2.minMaxLoc(match_result)[1]))

            stitch_json["ori_{}_{}.tif".format(index_y, index_x)] = [real_loc.tolist(), 0]  # [[y, x], error_num]
            # 覆写mask
            # 先丢弃部分因仿射变换带来的黑边
            # crop_rate = 0.003
            crop_rate = 0
            img_crop = img[int(img.shape[0] * crop_rate): img.shape[0]-int(img.shape[0] * crop_rate),
                           int(img.shape[1] * crop_rate): img.shape[1]-int(img.shape[1] * crop_rate)]
            real_loc_crop = real_loc + np.asarray([int(img.shape[1] * crop_rate), int(img.shape[0] * crop_rate)])
            stitch_img[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                       real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = img_crop
            temp_circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                              real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = \
                cv2.absdiff(img_crop[..., conf.stitch_channal],
                            temp_circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                                              real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]]
                            )
            cv2.putText(temp_circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                                          real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]],
                        "ori_{}_{}.tif".format(index_y, index_x), (200, 200), 0, 3, 255,
                        thickness=2)

        # 添加stitch的自校正
        save_json(os.path.join(save_dir, "stitch_json_ori.json"), stitch_json)
    else:
        print("stitch_json_ori.json is exists, load it")
        stitch_json = load_json(os.path.join(save_dir, "stitch_json_ori.json"))

    # 所有视野位置自校正
    new_stitch_json, fit_model = auto_correct_stitch_json(stitch_json, template_mask=std_circle.circles_mask)

    save_json(os.path.join(save_dir, "stitch_json.json"), new_stitch_json)
    # return new_stitch_json

    print("start save {}".format(os.path.join(save_dir, r"new_stitch_img_mask.tif")))
    tifffile.imwrite(os.path.join(save_dir, "new_stitch_img_mask.tif"), temp_circles_mask[::3, ::3, ...],
                     compression="jpeg")
    # print("start save {}".format(os.path.join(save_dir, r"new_stitch_img.tif")))
    # tifffile.imwrite(os.path.join(save_dir, "new_stitch_img.tif"),
    #                  stitch_img[std_circle.reg_box[0][1]:std_circle.reg_box[1][1],
    #                             std_circle.reg_box[0][0]:std_circle.reg_box[1][0]],
    #                  compression="jpeg")

    # 自校正后，根据新的json重新画图
    stitch_img = draw_img_by_json(pics_dir, os.path.join(save_dir, "stitch_json.json"), save_dir)

    ### 调试代码###
    # stitch_img = binary_pic(stitch_img)
    # std_circle.circles_img = cv2.warpPerspective(std_circle.circles_img, np.asarray(M),
    #                                              (std_circle.circles_img.shape[1], std_circle.circles_img.shape[0]))
    # std_circle_reg_box = stitch_json["std_reg_box"]
    # tmp_circles_img = std_circle.circles_img[std_circle_reg_box[0][1]:std_circle_reg_box[1][1],
    #                                          std_circle_reg_box[0][0]:std_circle_reg_box[1][0]]
    # stitch_img[..., 1] += tmp_circles_img
    stitch_img[std_circle.circles_img > 50] = [125, 125, 125]

    # 绘制双线性拟合点位与实际匹配点位
    tmp_all_index = np.array([x for x in stitch_json["all_index_y_x"] if f"ori_{x[0]}_{x[1]}.tif" in stitch_json])
    for tmp_index in tmp_all_index:
        tmp_y, tmp_x = tmp_index
        tmp_pic_name = f"ori_{tmp_y}_{tmp_x}.tif"
        ori_loc = stitch_json[tmp_pic_name][0]
        pred_loc = fit_model.predict(np.array([[tmp_y, tmp_x]]))[0].astype(int).tolist()
        cv2.circle(stitch_img, (pred_loc[0], pred_loc[1]), 10, (255, 255, 255), -1)  # 先画白色的预测点位
        cv2.circle(stitch_img, (ori_loc[0], ori_loc[1]), 10, (0, 0, 255), -1)  # 先画蓝色的，校正前的实际点位
        # cv2.putText(stitch_img, tmp_pic_name, 0, 1, (0, 0, 255),
        #             thickness=4)


    tifffile.imwrite(os.path.join(save_dir, "new_stitch_img_with_circles.tif"), stitch_img, compression="jpeg")

    return os.path.join(save_dir, "img_dist_with_board.tif")


def auto_correct_stitch_json(stitch_json, template_mask=None):
    # 坐标采用双线性拟合的方法，过滤出错误值较大的数据，用拟合数据代替。达到自校正效果
    # stitch_json = load_json(r"E:\Cell_seg_images\20230417-BK!6BN01F1-B3-SN-FG20-GE-20X-Cut-new_stitch\stitch_json.json")
    [start_index_x, start_index_y], [end_index_x, end_index_y], = stitch_json["x_y_range"]

    X = np.array([x for x in stitch_json["all_index_y_x"] if f"ori_{x[0]}_{x[1]}.tif" in stitch_json])
    Y = np.asarray([stitch_json[f"ori_{x[0]}_{x[1]}.tif"][0] for x in stitch_json["all_index_y_x"]
                    if f"ori_{x[0]}_{x[1]}.tif" in stitch_json])

    model = LinearRegression()
    model.fit(X, Y)

    # 做一次数据筛选，剔除误差最大的20%
    filter_num = len(X) // 5
    error_bar = np.sum(np.abs(Y - model.predict(X)), axis=1)
    filter_index = np.argsort(error_bar)[-filter_num:]
    X_filtered = np.delete(X, filter_index, axis=0)
    y_filtered = np.delete(Y, filter_index, axis=0)

    model = LinearRegression()
    model.fit(X_filtered, y_filtered)

    # 开始依次评估误差指数
    for y_i, x_i in stitch_json["all_index_y_x"]:
        pic_name = f"ori_{y_i}_{x_i}.tif"
        if pic_name not in stitch_json:
            real_loc = [-1, -1]
        else:
            real_loc = stitch_json[pic_name][0]
        pred_loc = model.predict(np.array([[y_i, x_i]]))[0].astype(int).tolist()
        error_num = abs(pred_loc[0] - real_loc[0]) + abs(pred_loc[1] - real_loc[1])

        print(f"pic_name: {pic_name}    real_loc: {real_loc}    pred_loc: {pred_loc}    error: {error_num}")
        if error_num > 1.5 * int(conf.conf.get("match-imgs", "match_range")):
            if template_mask is not None:
                img_ori = tifffile.imread(os.path.join(stitch_json["pics_dir"], pic_name))
                img_merge = cv2.cvtColor(img_ori[..., conf.stitch_channal], cv2.COLOR_GRAY2BGR)
                # img_merge = cv2.cvtColor(img_ori[..., 0] | img_ori[..., 1] | img_ori[..., 2], cv2.COLOR_GRAY2BGR)

                ###添加匹配过程###
                match_range = int(conf.conf.get("match-imgs", "match_range"))
                # match_range = 1
                # match_range = 1
                template = template_mask[
                           pred_loc[1] - match_range:pred_loc[1] + img_ori.shape[0] + match_range,
                           pred_loc[0] - match_range:pred_loc[0] + img_ori.shape[1] + match_range]
                if template.shape != tuple(np.asarray(img_merge.shape[:2]) + match_range * 2):
                    print("Different shape! template shape:{}, img shape shape:{}".format(template.shape, img_merge.shape[:2]))
                    continue
                match_result = cv2.matchTemplate(template, img_merge[..., conf.stitch_channal],
                                                 cv2.TM_CCOEFF)
                # match_shift = cv2.minMaxLoc(match_result)[2] - np.asarray([match_range, match_range])
                match_shift = cv2.minMaxLoc(match_result)[3] - np.asarray([match_range, match_range])
                pred_loc = (pred_loc + match_shift).tolist()

                ori_pred_loc = model.predict(np.array([[y_i, x_i]]))[0].astype(int).tolist()
                error_num = abs(pred_loc[0] - ori_pred_loc[0]) + abs(pred_loc[1] - ori_pred_loc[1])
                print(f"{pic_name} may be wrong, match_shift {match_shift}    error: {error_num}")
                ################
                #调试
                # img = cv2.copyMakeBorder(img, match_range, match_range, match_range, match_range, cv2.BORDER_CONSTANT, value=0)
                # show_img(cv2.merge([template, img[...,2], template]))
                # a = cv2.bitwise_not(img_merge[..., conf.stitch_channal])
                # b = template[40:40 + a.shape[0], 40:40 + a.shape[1]]
                # show_img(a)
                # show_img(b)
                # diff = cv2.absdiff(b, a)
                # show_img(diff)
                # squared_diff = cv2.multiply(diff, diff)
                # show_img(squared_diff)
                #
                # template = template_mask[
                #            pred_loc[1] - match_range:pred_loc[1] + img_ori.shape[0] + match_range,
                #            pred_loc[0] - match_range:pred_loc[0] + img_ori.shape[1] + match_range]
                # show_img(cv2.merge([template, img[...,2], template]))
                #####

            stitch_json[pic_name] = [pred_loc, error_num]
            print(f"{pic_name} may be wrong, correct it to {str(pred_loc)}")

        stitch_json[pic_name][1] = error_num
    return stitch_json, model


# 对CHIP的四个角坐标进行排序
# 排序后结果举例：[[3064,1800],[53160,1728],[53240,52144],[3136,52216]]
def sort_reg_box(ori_box):
    # 先把第一个元素小的放前两个
    sort_0 = sorted(ori_box)
    # 按照第2个元素排序
    tmp_sort_0 = sorted(sort_0[:2], key=lambda x: x[1])
    tmp_sort_1 = sorted(sort_0[2:], key=lambda x: x[1])

    sort_box = [tmp_sort_0[0], tmp_sort_1[0], tmp_sort_1[1], tmp_sort_0[1]]
    return sort_box


def cut_and_stitch(mrxs_path, reg_box):
    reg_box = np.array(reg_box)
    # 使用OpenCV的minAreaRect函数来找到最小外接矩形
    rect = cv2.minAreaRect(reg_box)
    # 获取矩形的四个角点
    box = cv2.boxPoints(rect)
    box = np.int0(box).tolist()

    reg_box = sort_reg_box(box)

    if mrxs_path.endswith(".mrxs"):
        pic_dir = mrxs_path.replace(".mrxs", "-Cut")
        fov_shape, pic_shape = cut_fov_img_S2000(mrxs_path, pic_dir, conf.camera_resolution)
        # if "S2000" in conf.base_mode or "S3000" in conf.base_mode:
        #     fov_shape, pic_shape = cut_fov_img_S2000(mrxs_path, pic_dir, conf.camera_resolution)
        # else:
        #     fov_shape, pic_shape = cut_fov_img(mrxs_path, pic_dir, conf.camera_resolution)
    elif mrxs_path.endswith('.svs'):
        pic_dir = mrxs_path.replace(".svs", "-Cut")
        fov_shape, pic_shape = cut_fov_img_haidexing(mrxs_path, pic_dir, conf.camera_resolution)
    elif mrxs_path.endswith('.tif'):
        pic_dir = mrxs_path.replace(".tif", "-Cut")
        fov_shape, pic_shape = cut_fov_img_tif(mrxs_path, pic_dir, conf.camera_resolution)
    print("End cut pic, start stitch pic...")

    save_dir = pic_dir + '-new_stitch'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    stitch_pic_path = new_stitch(pic_dir, reg_box, pic_shape=pic_shape, save_dir=save_dir)

    return stitch_pic_path


def draw_img_by_json(pics_dir, stitch_json_path, save_dir):
    stitch_json = load_json(stitch_json_path)
    M = stitch_json["M"]
    stitch_img = np.zeros((*conf.whole_img_size, 3), dtype=np.uint8)
    print("start match img---")
    x_y_range, all_index_y_x = stitch_json["x_y_range"], stitch_json["all_index_y_x"]
    for index_y_x in all_index_y_x:
        index_y, index_x = index_y_x
        if "ori_{}_{}.tif".format(index_y, index_x) not in stitch_json:
            print("ori_{}_{}.tif not have loc, skip it...".format(index_y, index_x))
            continue
        real_loc = stitch_json["ori_{}_{}.tif".format(index_y, index_x)][0]
        print("ori_{}_{}.tif : {}".format(index_y, index_x, real_loc))

        # img_ori = cv2.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))
        img_ori = tifffile.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))[..., :3]
        # img = cv2.warpPerspective(img_ori, np.asarray(M), img_ori.shape[:2][::-1], borderMode=cv2.BORDER_REFLECT_101)
        # img = cv2.warpPerspective(img_ori, np.asarray(M), img_ori.shape[:2][::-1])
        # img = cv2.warpPerspective(img_ori, np.asarray(M), (int(img_ori.shape[1] * 1.01), int(img_ori.shape[0] * 1.01)), borderValue=[0, 0, 0])

        border_ex = 0
        # img = cv2.warpPerspective(img_ori, np.eye(3), (img_ori.shape[1]+border_ex, img_ori.shape[0]+border_ex), borderMode=cv2.BORDER_REPLICATE)  # 不做变形，只稍微扩充边界，防止黑边
        img = cv2.copyMakeBorder(img_ori, border_ex, border_ex, border_ex, border_ex, cv2.BORDER_REPLICATE)  # 不做变形，只稍微扩充边界，防止黑边
        # img = img_ori
        # 先丢弃部分因仿射变换带来的黑边
        # crop_rate = 0.003
        crop_rate = 0
        img_crop = img[int(img.shape[0] * crop_rate): img.shape[0]-int(img.shape[0] * crop_rate),
                       int(img.shape[1] * crop_rate): img.shape[1]-int(img.shape[1] * crop_rate)]
        # img_crop = binary_pic(img_crop)

        real_loc_crop = real_loc + np.asarray([int(img.shape[1] * crop_rate), int(img.shape[0] * crop_rate)]) - border_ex
        sub_stitch = stitch_img[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                                real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]]
        # tmp_mask = np.where(sub_stitch == [0, 0, 0], 255, 0)
        # sub_stitch[:, :, :] = np.where(tmp_mask, img_crop, sub_stitch)
        # show_img(sub_stitch)
        sub_stitch[:, :, :] = merge_two_img(sub_stitch, img_crop)
        # sub_stitch[:, :, :] = scipy.ndimage.rotate(img_crop, -0.3)
        # show_img(sub_stitch)
        # sub_stitch[:, :, :] = cv2.addWeighted(sub_stitch, 1, img_crop, 1, 0)

    print("start save {}".format(os.path.join(save_dir, r"new_stitch_img.tif")))

    # stitch_img = binary_pic(stitch_img)

    # tifffile.imwrite(os.path.join(save_dir, "new_stitch_img.tif"), stitch_img, compression="jpeg")
    chip_stitch_img = my_warpPerspective(stitch_img, np.asarray(M), (stitch_img.shape[1], stitch_img.shape[0]))
    #
    std_circle_reg_box = stitch_json["std_reg_box"]
    img_dist = chip_stitch_img[std_circle_reg_box[0][1]:std_circle_reg_box[1][1],
                               std_circle_reg_box[0][0]:std_circle_reg_box[1][0]]

    if img_dist.shape[0] > conf.out_size:
        out_size_rate_w_h = ((conf.barcode_size_x+1)*2*conf.base_size_x)/((conf.barcode_size_y+1)*np.sqrt(3)*conf.base_size_y)
        img_dist = cv2.resize(img_dist, (int(conf.out_size * out_size_rate_w_h), conf.out_size))

    tifffile.imwrite(os.path.join(save_dir, "img_dist.tif"),
                     img_dist,
                     compression="LZW")

    # 画底板
    from need.CorrectWholeImg import cal_zoom_rate, gen_std_board_loc, gen_std_board_img
    zoom_scale = cal_zoom_rate(img_dist.shape[1], img_dist.shape[0])
    std_kp_loc, std_w_h = gen_std_board_loc(zoom_scale)
    img_dist_with_board = gen_std_board_img(img_dist.shape[1], img_dist.shape[0], std_kp_loc, save_dir,
                                            base_img=img_dist, mask_color=conf.std_mask_color)
    tifffile.imwrite(os.path.join(save_dir, "img_dist_with_board.tif"), img_dist_with_board,
                     compression=conf.compression_mode)
    ##

    return stitch_img


def find_distance(img, M=None, micro_rate=10):
    # 利用寻峰方法找到一张二值化图的平均分界线距离
    if M is not None:
        img = my_warpPerspective(img, M, img.shape[:2][::-1])

    img_b = img[..., conf.stitch_channal]
    # show_img(img_b)

    x_sum = np.sum(img_b, axis=0, dtype=int)

    # x_sum = np.interp(np.linspace(0, len(x_sum)-1, 10000), np.arange(len(x_sum)), x_sum)

    fft_signal = abs(np.fft.fft(x_sum, n=len(x_sum)*micro_rate))
    # 控制数据范围，避免离谱数据
    fft_signal[0] = 0
    fft_signal[:len(fft_signal)//20] = 0
    fft_signal[-len(fft_signal)//2:] = 0

    # import matplotlib.pyplot as plt
    # plt.plot(fft_signal)
    # plt.plot(x_sum)
    f = np.argmax(fft_signal) / micro_rate

    # show_img(img_b)
    # plt.close()
    return len(x_sum)/f * 2  # x方向统计纵轴频率间距，由于微球是错行相隔，所以频率会翻倍，所以计算“周期”要*2


def calculate_std_d(pics_dir, stitch_json):
    M = np.array(stitch_json["M"])
    x_y_range = stitch_json["x_y_range"]

    all_mean_distance = []
    for i in range(20):
        # 随机选择序号，避开边缘
        random_x = random.randint(x_y_range[0][0] + 1, x_y_range[1][0] - 1)
        random_y = random.randint(x_y_range[0][1] + 1, x_y_range[1][1] - 1)
        img = tifffile.imread(os.path.join(pics_dir, f"ori_{random_y}_{random_x}.tif"))
        mean_distance = find_distance(img, M=M)
        all_mean_distance.append(mean_distance)
        print(f"ori_{random_y}_{random_x}.tif -> mean_distance:{mean_distance}")
    return np.median(all_mean_distance)


def find_error_pic(error_point, stitch_json):
    pic_shape = stitch_json["pic_shape"]
    # 遍历cycle_json中每一个视野的位置，定位error位置的视野序号
    error_point_x, error_point_y = error_point
    for index_y_x in stitch_json["all_index_y_x"]:
        img_name = f"ori_{index_y_x[0]}_{index_y_x[1]}.tif"
        left_top_loc = stitch_json[img_name][0]
        if left_top_loc[0] <= error_point_x <= left_top_loc[0] + pic_shape[1] and \
                left_top_loc[1] <= error_point_y <= left_top_loc[1] + pic_shape[0]:
            return img_name, left_top_loc
    raise Exception("Error: Can't find error pic by error point:{}. Check it please!".format(error_point))


def draw_part_circle_pic(circle_centers, stitch_json, start_loc, shape=(conf.barcode_size_x, conf.barcode_size_y)):
    std_d = stitch_json["std_circles_d"]
    std_r = int(std_d * 0.4)

    pic_shape = stitch_json["pic_shape"]
    start_loc = np.asarray(start_loc)
    end_loc = start_loc + pic_shape[::-1]
    # 找到开始位置和结束位置最接近的圆心坐标
    distance_values = np.sum(np.abs(circle_centers - start_loc), axis=2)
    x_i_start, y_i_start = cv2.minMaxLoc(distance_values)[2][::-1]

    distance_values = np.sum(np.abs(circle_centers - end_loc), axis=2)
    x_i_end, y_i_end = cv2.minMaxLoc(distance_values)[2][::-1]

    circle_img = np.zeros((*pic_shape, 3), dtype=np.uint8)
    for x in range(x_i_start, x_i_end+1):
        for y in range(y_i_start, y_i_end+1):
            if y % (shape[1] + 1) == 0 or x % (shape[0] + 1) == 0:
                # 不同块的分割线
                continue
            offset_loc = circle_centers[x, y] - start_loc
            cv2.circle(circle_img, tuple(offset_loc), std_r, (155, 155, 155), 1)
    return circle_img


def correct_img(wrong_point_norm, stitch_json_path):
    stitch_json = load_json(stitch_json_path)
    pic_dir = stitch_json["pics_dir"]
    std_reg_box = stitch_json["std_reg_box"]
    reg_box = stitch_json["reg_box"]
    x_width, y_width = (std_reg_box[1][0] - std_reg_box[0][0],
                        std_reg_box[1][1] - std_reg_box[0][1])
    # 首先，根据wrong_point的相对位置和decode_img的标准框区域，确定绝对坐标
    ori_point = (wrong_point_norm[0] * x_width + std_reg_box[0][0],
                 wrong_point_norm[1] * y_width + std_reg_box[0][1])
    # 然后根据M将绝对坐标逆变换成实际坐标
    M = stitch_json["M"]
    M_inv = np.linalg.inv(M)
    real_point = np.round(np.dot(M_inv, np.array([*ori_point, 1]))).astype(int)[:2]
    # 找到错误图像
    error_img_name, error_img_loc = find_error_pic(real_point, stitch_json)
    error_img = tifffile.imread(os.path.join(pic_dir, error_img_name))[..., conf.stitch_channal]
    error_img = cv2.cvtColor(error_img, cv2.COLOR_GRAY2RGB)
    circle_centers = np.load(stitch_json["std_circles_center_path"])
    circle_img = draw_part_circle_pic(circle_centers, stitch_json, error_img_loc, shape=(conf.barcode_size_x, conf.barcode_size_y))
    # add_img = np.where(circle_img > 0, circle_img, error_img)
    # show_img(add_img[:500, :500])
    new_shift = shift_move_show_img(circle_img, error_img, scale_size=(800, 800),
                                    name=f"correct"
                                    , center_point=real_point - error_img_loc)
    # tifffile.imwrite(r"E:\test\S3000\S3000-20X-MC-Cut-new_stitch\circle.tif", circle_img)
    # tifffile.imwrite(r"E:\test\S3000\S3000-20X-MC-Cut-new_stitch\sub.tif", error_img)
    if new_shift and new_shift != [0, 0]:
        correct_loc = error_img_loc + np.asarray(new_shift, dtype=int)
        # 更新stitch_json.json
        stitch_json[error_img_name][0] = correct_loc.tolist()
        stitch_json[error_img_name][1] = 0
        save_json(stitch_json_path, stitch_json)
        print(f"correct pic :{error_img_name}, local:{error_img_loc}->{correct_loc.tolist()}")


if __name__ == '__main__':
    pass
    # pic_dir = r"E:\test\no_new.tif"
    # reg_box = [[2478, 3924], [43542, 3926], [43540, 44712], [2478, 44702]]
    # stitch_json = cut_and_stitch(pic_dir, reg_box)

    pics_dir = r"E:\test\slide-2024-06-04T10-36-24-R1-S3-Cut"
    stitch_json = load_json(r"E:\test\slide-2024-06-04T10-36-24-R1-S3-Cut-new_stitch\stitch_json.json")
    a = calculate_std_d(pics_dir, stitch_json)

    # img=r"E:\test\0129-20231115-CA31BN02F5-B4-PCA2-20XFL-20240129-154003222-Cut\ori_2_9.tif"
    # img = tifffile.imread(img)
    # a = find_distance(img)
