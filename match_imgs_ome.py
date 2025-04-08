import os.path
# import time
#
# import zarr
import time
import traceback

import cv2
import numpy as np
import random
# # import openslide
# import scipy
# import skimage.transform
# import tifffile
import tifffile

from need.ofen_tool import save_json, load_json, shift_move_show_img, show_img
from need.KpDetectByYolo import MyDetector
from need.config import conf
# from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

import platform
if platform.system() == 'Windows':
    vipsbin = r'D:\work\python\vips-dev-8.15\bin'
    os.environ['PATH'] = r"./src/vips-dev-8.15/bin" + ';' + vipsbin + ';' + os.environ['PATH']
import pyvips

import need.BmTiffLib as BmTiff

detector = MyDetector("./model/best.onnx")
# std_edge_size = (2248, 2648)


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
    # 进行加权平均
    alpha = 0.5  # 权重，范围在 [0, 1] 之间
    blended_image = (a * alpha + b * (1 - alpha)).cast('uchar')
    return blended_image

class StdCircles:
    def __init__(self, ori_size, shape, d, r, rot=None):

        # import pdb;pdb.set_trace()
        self.circles_img, self.circles_mask, self.circle_centers, self.reg_box = \
            self.std_circles(ori_size, shape, d, r, tile_limit=(conf.base_size_x, conf.base_size_y))

        # BmTiff.save_pyramid_tif(self.circles_mask, r"E:\test\A_big_tiff\merge\circles_mask.ome.tif")

        # 如果存在rot，则做一下变换vim
        if rot is not None:  # 反着转个角度，使其在缝合过程避免变换丢失数据，最后再变换回正确图案
            self.circles_img = BmTiff.rotate_and_cinter_crop(self.circles_img, -rot)
            self.circles_mask = BmTiff.rotate_and_cinter_crop(self.circles_mask, -rot)

            center = (self.circles_mask.width / 2, self.circles_mask.height / 2)
            rotation_matrix_2x3 = cv2.getRotationMatrix2D(center, -rot, 1.0)
            rotation_matrix_3x3 = np.vstack([rotation_matrix_2x3, [0, 0, 1]])
            M_inv = np.linalg.inv(rotation_matrix_3x3)
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
        # cv2.imwrite("test_circles_mask.tif", self.circles_mask)
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
        # self.std_mask = self.circles_img[self.std_first_div_point[0]:self.std_first_div_point[0] + self.std_mask_size[0],
        #                                  self.std_first_div_point[1]:self.std_first_div_point[1] + self.std_mask_size[1]]
        # self.std_mask_centers = self.circle_centers[conf.barcode_size_x + 2: conf.barcode_size_x + 2 + conf.barcode_size_x,
        #                                             conf.barcode_size_y + 2: conf.barcode_size_y + 2 + conf.barcode_size_y] - self.std_first_div_point[::-1]


    @staticmethod
    def std_circles(ori_size, shape, d, r, tile_limit=(46, 46), M=None):
        # std_circles((1200, 1200), (30, 35), 35, 10)
        size = (ori_size[0] + 30, ori_size[1] + 30)
        # circles_img = BmTiff.draw_black(size[0], size[1], bands=1)
        # circles_mask = BmTiff.draw_black(size[0], size[1], bands=1)
        circles_img = np.zeros(size[::-1], dtype=np.uint8)
        circles_mask = np.zeros(size[::-1], dtype=np.uint8)
        delta_x = d
        delta_y = d * 0.5 * 3 ** 0.5
        # if conf.base_mode == "huge":
        #     delta_y = delta_x * 0.8531320425180289  # 大芯片的高度不是标准的三角形
        total_circles_shape = (int(size[0] / delta_x) + 1, int(size[1] / delta_y) + 1)
        # first_circle_loc = (np.asarray(size) - (np.asarray(shape) - 1) * np.asarray((delta_x, delta_y))) // 2
        # first_circle_loc = np.asarray((delta_x, delta_y), dtype=int)
        first_circle_loc = np.asarray((conf.std_edge_size[1], conf.std_edge_size[0]), dtype=int)

        x_indices, y_indices = np.meshgrid(np.arange(total_circles_shape[0]), np.arange(total_circles_shape[1]),
                                           indexing='ij')
        # 偶数行圆心坐标索引
        first_loc = first_circle_loc - (d / 2, 0)
        circle_centers_even = np.stack((x_indices, y_indices), axis=-1).astype(float)
        circle_centers_even *= (delta_x, delta_y)
        circle_centers_even += first_loc
        circle_centers_even = circle_centers_even.astype(int)

        # 奇数行圆心坐标索引
        first_loc = first_circle_loc
        circle_centers_odd = np.stack((x_indices, y_indices), axis=-1).astype(float)
        circle_centers_odd *= (delta_x, delta_y)
        circle_centers_odd += first_loc
        circle_centers_odd = circle_centers_odd.astype(int)

        for y in range(total_circles_shape[1]):
            if y % 2 == 0:
                circle_centers_odd[:, y, :] = circle_centers_even[:, y, :]
        circle_centers = circle_centers_odd
        y_limit = (shape[1] + 1) * tile_limit[1] + 1
        x_limit = (shape[0] + 1) * tile_limit[0] + 1
        circle_centers = circle_centers[:x_limit, :y_limit]
        # np.save(r"E:\test\A_big_tiff\merge\circle_centers.npy", circle_centers_odd)
        # circle_centers = np.zeros((*total_circles_shape, 2), dtype=int)
        # for y in range(total_circles_shape[1]):
        #     for x in range(total_circles_shape[0]):
        #
        #         first_loc = first_circle_loc
        #         if y % 2 == 0:
        #             first_loc = first_circle_loc - (d / 2, 0)
        #         tmp_center = (first_loc + (x * delta_x, y * delta_y)).astype(int)
        #         # circle_centers.append(tmp_center.astype(int))
        #         circle_centers[x, y, :] = tmp_center

        reg_box = [circle_centers[0, 0].tolist(),
                   circle_centers[(conf.barcode_size_x + 1) * conf.base_size_x,
                                  (conf.barcode_size_y + 1) * conf.base_size_y].tolist()]
        reg_box[0][0] += round(d//2)
        reg_box[1][0] += round(d//2)

        t1 = time.time()
        x_coords = circle_centers[:, :, 0]
        y_coords = circle_centers[:, :, 1]
        circles_mask[y_coords, x_coords] = int(255 * 0.8)
        circles_img[y_coords, x_coords] = 255

        # # 对边缘最近的一行的权重提高，这样在计算时候可以让匹配尽量贴近边缘，避免边缘不完整图像匹配出错
        for y in range(0, circle_centers.shape[1], (shape[1] + 1)):

            x_coords = circle_centers[:, y-1, 0]
            y_coords = circle_centers[:, y-1, 1]
            circles_mask[y_coords, x_coords] = 255

            if y + 1 >= circle_centers.shape[1]:
                continue
            x_coords = circle_centers[:, y+1, 0]
            y_coords = circle_centers[:, y+1, 1]
            circles_mask[y_coords, x_coords] = 255
        for x in range(0, circle_centers.shape[0], (shape[0] + 1)):
            x_coords = circle_centers[x - 1, :, 0]
            y_coords = circle_centers[x - 1, :, 1]
            circles_mask[y_coords, x_coords] = 255

            if x + 1 >= circle_centers.shape[0]:
                continue

            x_coords = circle_centers[x + 1, :, 0]
            y_coords = circle_centers[x + 1, :, 1]
            circles_mask[y_coords, x_coords] = 255
        # 不同块的分割线
        for y in range(0, circle_centers.shape[1], (shape[1] + 1)):
            x_coords = circle_centers[:, y, 0]
            y_coords = circle_centers[:, y, 1]
            circles_mask[y_coords, x_coords] = 0
            circles_img[y_coords, x_coords] = 0
        for x in range(0, circle_centers.shape[0], (shape[0] + 1)):
            x_coords = circle_centers[x, :, 0]
            y_coords = circle_centers[x, :, 1]
            circles_mask[y_coords, x_coords] = 0
            circles_img[y_coords, x_coords] = 0
            # print(y)


        print(f"Draw std circles cost:{time.time()-t1}")

        mask_kernel_size = (r+r)*2-1  # 调整这个值改变膨胀的强度，膨胀成圆形
        mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_kernel_size, mask_kernel_size))
        circles_mask = cv2.dilate(circles_mask, mask_kernel, iterations=1)

        # circles__kernel_size = (r+1)*2  # 调整这个值改变膨胀的强度，膨胀成圆形
        circles__kernel_size = 3  # 调整这个值改变膨胀的强度，膨胀成圆形
        circles__kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circles__kernel_size, circles__kernel_size))
        circles_img_out = cv2.dilate(circles_img, circles__kernel, iterations=1)
        # circles__kernel_size = 3  # 腐蚀一层，然后再减去
        # circles__kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circles__kernel_size, circles__kernel_size))
        # circles_img_in = cv2.erode(circles_img_out, circles__kernel, iterations=1)
        # circles_img = circles_img_out - circles_img_in
        circles_img = circles_img_out

        circles_mask = cv2.erode(circles_mask, cv2.getStructuringElement(cv2.MORPH_ERODE, (r-1, r-1)))
        circles_mask = cv2.bitwise_not(circles_mask)


        circles_img = pyvips.Image.new_from_array(circles_img[:ori_size[1], :ori_size[0]])
        circles_mask = pyvips.Image.new_from_array(circles_mask[:ori_size[1], :ori_size[0]])

        return [circles_img,
                circles_mask,
                circle_centers,
                reg_box]


def get_fov_info(pyramid_img_path, camera_resolution=None):

    whole_img = BmTiff.read_pyramid_from_file(pyramid_img_path, page=conf.stitch_channal)
    tiles_width = int(BmTiff.get_property_value(whole_img, "tiles_width"))
    tiles_height = int(BmTiff.get_property_value(whole_img, "tiles_height"))
    tiles_n_across = int(BmTiff.get_property_value(whole_img, "tiles_n_across"))
    tiles_n_down = int(BmTiff.get_property_value(whole_img, "tiles_n_down"))
    FOV_PIXES_x_y = (tiles_width, tiles_height)
    FOV_SHAPE_x_y = (tiles_n_across, tiles_n_down)

    return FOV_SHAPE_x_y, FOV_PIXES_x_y


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


def gen_index_by_reg_box(reg_box, FOV_PIXES_x_y):
    reg_box = np.asarray(reg_box)
    x_start = min(reg_box[0][0], reg_box[3][0]) // FOV_PIXES_x_y[0]
    y_start = min(reg_box[0][1], reg_box[1][1]) // FOV_PIXES_x_y[1]

    x_end = (max(reg_box[2][0], reg_box[1][0])-1) // FOV_PIXES_x_y[0]
    y_end = (max(reg_box[2][1], reg_box[3][1])-1) // FOV_PIXES_x_y[1]

    all_index_x_y = []
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            all_index_x_y.append([x, y])
    return [[int(x_start), int(y_start)], [int(x_end), int(y_end)]], all_index_x_y


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
    rotation_matrix_2x3 = cv2.getRotationMatrix2D(center, -angle, 1.0)

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


def caculate_angle_M(stitch_json):
    pyramid_img_path = stitch_json["pyramid_img_path"]
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]
    x_y_range = stitch_json["x_y_range"]

    all_angle = []
    all_M = []
    whole_img = BmTiff.read_pyramid_from_file(pyramid_img_path, page=conf.stitch_channal)
    for i in range(5):
        # 随机选择序号，避开边缘
        random_x = random.randint(x_y_range[0][0] + 1, x_y_range[1][0] - 1)
        random_y = random.randint(x_y_range[0][1] + 1, x_y_range[1][1] - 1)
        crop_range = (random_x * FOV_PIXES_x_y[0], random_y * FOV_PIXES_x_y[1],
                      FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
        img = whole_img.crop(*crop_range).numpy()
        best_angle, best_rot_matrix = find_best_rotate(img)
        all_angle.append(best_angle)
        all_M.append(best_rot_matrix)
        print(f"crop:{crop_range} -> best_angle:{best_angle}")
    all_angle = np.array(all_angle)
    median_angle = np.median(all_angle)
    median_index = np.where(all_angle == median_angle)[0]  # 取中位数索引
    median_M = all_M[median_index[0]]
    return median_angle, median_M


def new_stitch(pyramid_img_path, reg_box, FOV_PIXES_x_y=(2048, 2448), save_dir=None):
    if save_dir is None:
        save_dir = pyramid_img_path.replace(".ome.tif", "")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, "stitch_json.json")):
        reg_box = np.asarray(reg_box)
        stitch_json = {"reg_box": reg_box.tolist(),
                       "FOV_PIXES_x_y": np.asarray(FOV_PIXES_x_y, dtype=int).tolist(),
                       "pyramid_img_path": pyramid_img_path}

        x_y_range, all_index_x_y = gen_index_by_reg_box(reg_box, FOV_PIXES_x_y)
        stitch_json["x_y_range"], stitch_json["FOV_PIXES_x_y"] = x_y_range, FOV_PIXES_x_y
        stitch_json["all_index_x_y"] = all_index_x_y

        # 估算图像偏转角度

        angle, M = caculate_angle_M(stitch_json)
        print("final angle: {}".format(angle))
        stitch_json["angle"] = float(angle)

        stitch_json["M"] = M.tolist()
        # std_d = 14.83
        # std_d = 14.836363636363636
        std_d = calculate_std_d(stitch_json)

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
        whole_img_size = (int(conf.std_edge_size[0] * 2 + std_d * (conf.barcode_size_x + 1) * conf.base_size_x) + 1,
                          int(conf.std_edge_size[1] * 2 + std_d * 0.5 * np.sqrt(3) * (conf.barcode_size_y + 1) * conf.base_size_y) + 1)
        conf.conf.set("match-imgs", "whole_img_size", str(whole_img_size))
        with open(conf.ini_path, "w", encoding="utf-8") as conf_ini:
            conf.conf.write(conf_ini)
        conf.whole_img_size = whole_img_size

        std_circle = StdCircles(conf.whole_img_size, (conf.barcode_size_x, conf.barcode_size_y), std_d, std_r, rot=angle)
        stitch_json["std_reg_box"] = std_circle.reg_box
        stitch_json["std_circles_d"] = std_d
        stitch_json["std_circles_center_path"] = os.path.join(save_dir, "circle_centers.npy")

        np.save(stitch_json["std_circles_center_path"], std_circle.circle_centers)
        BmTiff.save_pyramid_tif(std_circle.circles_mask, os.path.join(save_dir, "circle_mask.ome.tif"))

        print("---end draw std circles")
        # stitch_img = cv2.cvtColor(std_circle.circles_mask, cv2.COLOR_GRAY2BGR)
        # stitch_img = np.zeros((*conf.whole_img_size, 3), dtype=np.uint8)

        # whole_start_loc = reg_box[0]
        # whole_end_loc = reg_box[2]
        print("---end draw std stitch_img")
        print("start match img---")


        # 根据全图坐标和 reg_box位置，预测该关键点所在的区域,通过线性拟合后预测位置
        model = LinearRegression()
        X = np.array(reg_box)
        Y = np.asarray([[0, 0], [conf.base_size_x, 0], [conf.base_size_x, conf.base_size_y], [0, conf.base_size_y]])
        model.fit(X, Y)

        whole_img = BmTiff.read_pyramid_from_file(pyramid_img_path, page=conf.stitch_channal)
        if not os.path.exists(os.path.join(save_dir, "stitch_json_ori.json")):
            print("Start match stitch_json_ori.json")
            save_json(os.path.join(save_dir, "stitch_json_ori_bk.json"), stitch_json)
            for index_x_y in all_index_x_y:
                try:
                    index_x, index_y = index_x_y
                    # if not os.path.exists(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x))):
                    #     print("ori_{}_{}.tif is not exist".format(index_y, index_x))
                    #     continue
                    crop_range = (index_x * FOV_PIXES_x_y[0], index_y * FOV_PIXES_x_y[1],
                                  FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
                    img = whole_img.crop(*crop_range).numpy()
                    # img = cv2.warpPerspective(img_ori, M, img_ori.shape[:2][::-1], borderMode=cv2.BORDER_REFLECT_101)

                    # img = img_ori.copy()
                    # out_img, centers = detector.detect(img, 0.4)
                    img_merge = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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
                    # if conf.base_mode == "S2000-2":  # S2000-2减小尺寸做计算
                    #     match_range = match_range // 2
                    template = std_circle.circles_mask.crop(template_start_loc[0]-match_range, template_start_loc[1]-match_range,
                                                            img.shape[1]+2*match_range, img.shape[0]+2*match_range).numpy()
                    if template.shape != tuple(np.asarray(img.shape[:2]) + match_range * 2):
                        print("Different shape! template shape:{}, img shape shape:{}".format(template.shape, img.shape[:2]))
                        continue
                    # match_result = cv2.matchTemplate(template, cv2.bitwise_not(img_merge[..., conf.stitch_channal]), cv2.TM_SQDIFF)
                    # match_shift = cv2.minMaxLoc(match_result)[2] - np.asarray([match_range, match_range])
                    match_result = cv2.matchTemplate(template, img_merge[..., conf.stitch_channal], cv2.TM_CCOEFF)
                    match_shift = cv2.minMaxLoc(match_result)[3] - np.asarray([match_range, match_range])
                    real_loc = template_start_loc + match_shift

                    # ### 调试代码####
                    # if (index_y, index_x) == (3, 0):
                    #     tmp_show = np.zeros_like(img_merge)
                    #     tmp_show[..., 2] = img_merge[..., conf.stitch_channal]
                    #     # tmp_show[..., 1] = std_circle.circles_img[template_start_loc[1]:template_start_loc[1]+img_merge.shape[0],
                    #     #                                           template_start_loc[0]:template_start_loc[0]+img_merge.shape[1]]
                    #     # tmp_show[..., 0] = std_circle.circles_mask.crop(real_loc[0], real_loc[1], img_merge.shape[1], img_merge.shape[0]).numpy()
                    #     tmp_show[..., 0] = std_circle.circles_mask.crop(template_start_loc[0], template_start_loc[1], img_merge.shape[1], img_merge.shape[0]).numpy()
                    #     show_img(tmp_show)
                    #     import pdb;pdb.set_trace()
                    print("Match ori_{}_{}.tif, div_point:{}, shift:{}, match rate:{}"
                          "".format(index_y, index_x, max_match_kp, match_shift, cv2.minMaxLoc(match_result)[1]))

                    stitch_json["ori_{}_{}.tif".format(index_y, index_x)] = [real_loc.tolist(), 0]  # [[x, y], error_num]
                except Exception as e:
                    print("ori_{}_{}.tif, ERRRO".format(index_y, index_x))
                    # 打印错误堆栈
                    traceback.print_exc()
                    continue
                # 覆写mask
                # 先丢弃部分因仿射变换带来的黑边
                # # crop_rate = 0.003
                # crop_rate = 0
                # img_crop = img[int(img.shape[0] * crop_rate): img.shape[0]-int(img.shape[0] * crop_rate),
                #                int(img.shape[1] * crop_rate): img.shape[1]-int(img.shape[1] * crop_rate)]
                # real_loc_crop = real_loc + np.asarray([int(img.shape[1] * crop_rate), int(img.shape[0] * crop_rate)])
                # # stitch_img[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                # #            real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = img_crop
                # temp_circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                #                   real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = \
                #     cv2.absdiff(img_crop,
                #                 temp_circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                #                                   real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]]
                #                 )
                # cv2.putText(temp_circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                #                               real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]],
                #             "ori_{}_{}.tif".format(index_y, index_x), (200, 200), 0, 3, 255,
                #             thickness=2)

            # 添加stitch的自校正
            save_json(os.path.join(save_dir, "stitch_json_ori.json"), stitch_json)
        else:
            print("stitch_json_ori.json is exists, load it")
            stitch_json = load_json(os.path.join(save_dir, "stitch_json_ori.json"))

        # 所有视野位置自校正
        new_stitch_json, fit_model = auto_correct_stitch_json(stitch_json.copy(), template_mask=std_circle.circles_mask)
        # new_stitch_json = stitch_json
        save_json(os.path.join(save_dir, "stitch_json.json"), new_stitch_json)
        # return new_stitch_json

        print("start save {}".format(os.path.join(save_dir, r"new_stitch_img_mask.tif")))
        # tifffile.imwrite(os.path.join(save_dir, "new_stitch_img_mask.tif"), temp_circles_mask[::3, ::3, ...],
        #                  compression="jpeg")
        # print("start save {}".format(os.path.join(save_dir, r"new_stitch_img.tif")))
        # tifffile.imwrite(os.path.join(save_dir, "new_stitch_img.tif"),
        #                  stitch_img[std_circle.reg_box[0][1]:std_circle.reg_box[1][1],
        #                             std_circle.reg_box[0][0]:std_circle.reg_box[1][0]],
        #                  compression="jpeg")


    print("stitch_json is exists, Start stitch img by stitch_json.json")
    # 自校正后，根据新的json重新画图
    # stitch_img, stitch_img_crop = draw_img_by_json(os.path.join(save_dir, "stitch_json.json"), save_dir, page=conf.stitch_channal)
    stitch_img, stitch_img_crop = draw_img_by_json_numpy(os.path.join(save_dir, "stitch_json.json"), save_dir, page=conf.stitch_channal)

    if "std_circle" in locals():
        circles_img = std_circle.circles_img
        stitch_img = (stitch_img+circles_img).cast('uchar')
        BmTiff.save_pyramid_tif(stitch_img.copy(), os.path.join(save_dir, "stitch_img_with_circles.ome.tif"))

    # for channel in range(3):
    #     if channel != conf.stitch_channal:
    #         _, stitch_img_crop_other = draw_img_by_json(os.path.join(save_dir, "stitch_json.json"), save_dir,
    #                                                     page=channel)
    #         stitch_img_crop = stitch_img_crop.bandjoin(stitch_img_crop_other)

    BmTiff.save_pyramid_tif(stitch_img_crop, os.path.join(save_dir, "img_dist_merge.ome.tif"))

    # 下面保存一下目标图像的新尺寸tif
    # 目标高度
    out_size_rate_w_h = ((conf.barcode_size_x + 1) * 2 * conf.base_size_x) / (
                (conf.barcode_size_y + 1) * np.sqrt(3) * conf.base_size_y)
    new_size = (int(conf.out_size * out_size_rate_w_h), conf.out_size)

    BmTiff.save_special_size(os.path.join(save_dir, "img_dist_merge.ome.tif"), os.path.join(save_dir, "img_dist.tif"), new_size[0], new_size[1], compression="lzw")
    return os.path.join(save_dir, "img_dist_merge.ome.tif")


def auto_correct_stitch_json(stitch_json, template_mask=None):
    # 坐标采用双线性拟合的方法，过滤出错误值较大的数据，用拟合数据代替。达到自校正效果
    # stitch_json = load_json(r"E:\Cell_seg_images\20230417-BK!6BN01F1-B3-SN-FG20-GE-20X-Cut-new_stitch\stitch_json.json")
    [start_index_x, start_index_y], [end_index_x, end_index_y], = stitch_json["x_y_range"]
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]
    whole_img = pyvips.Image.new_from_file(stitch_json["pyramid_img_path"])

    X = np.array([x for x in stitch_json["all_index_x_y"] if f"ori_{x[1]}_{x[0]}.tif" in stitch_json])
    Y = np.asarray([stitch_json[f"ori_{x[1]}_{x[0]}.tif"][0] for x in stitch_json["all_index_x_y"]
                    if f"ori_{x[1]}_{x[0]}.tif" in stitch_json])

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
    for x_i, y_i in stitch_json["all_index_x_y"]:
        pic_name = f"ori_{y_i}_{x_i}.tif"
        if pic_name not in stitch_json:
            real_loc = [-1, -1]
        else:
            real_loc = stitch_json[pic_name][0]
        pred_loc = model.predict(np.array([[x_i, y_i]]))[0].astype(int).tolist()
        error_num = abs(pred_loc[0] - real_loc[0]) + abs(pred_loc[1] - real_loc[1])

        print(f"pic_name: {pic_name}    real_loc: {real_loc}    pred_loc: {pred_loc}    error: {error_num}")
        match_range = int(conf.conf.get("match-imgs", "match_range"))
        # if conf.base_mode == "S2000-2":  # S2000-2减小尺寸做计算
        #     match_range = match_range // 2
        if error_num > 1.5 * match_range:
            if template_mask is not None:
                crop_range = (x_i * FOV_PIXES_x_y[0], y_i * FOV_PIXES_x_y[1],
                              FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
                try:
                    img_merge = whole_img.crop(*crop_range).numpy()
                except Exception as e:
                    print("crop_range: {}, ERRRO".format(crop_range))
                    # 打印错误堆栈
                    traceback.print_exc()
                    continue
                img_merge = cv2.cvtColor(img_merge, cv2.COLOR_GRAY2BGR)
                # img_merge = cv2.cvtColor(img_ori[..., 0] | img_ori[..., 1] | img_ori[..., 2], cv2.COLOR_GRAY2BGR)

                ###添加匹配过程###
                # match_range = 1
                # match_range = 1
                template = template_mask.crop(pred_loc[0] - match_range, pred_loc[1] - match_range,
                                              img_merge.shape[1] + 2 * match_range,
                                              img_merge.shape[0] + 2 * match_range).numpy()
                if template.shape != tuple(np.asarray(img_merge.shape[:2]) + match_range * 2):
                    print("Different shape! template shape:{}, img shape shape:{}".format(template.shape, img_merge.shape[:2]))
                    continue
                match_result = cv2.matchTemplate(template, img_merge[..., conf.stitch_channal],
                                                 cv2.TM_CCOEFF)
                # match_shift = cv2.minMaxLoc(match_result)[2] - np.asarray([match_range, match_range])
                match_shift = cv2.minMaxLoc(match_result)[3] - np.asarray([match_range, match_range])
                pred_loc = (pred_loc + match_shift).tolist()

                ori_pred_loc = model.predict(np.array([[x_i, y_i]]))[0].astype(int).tolist()
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


def cut_and_stitch(pyramid_img_path, reg_box):
    reg_box = np.array(reg_box)
    # 使用OpenCV的minAreaRect函数来找到最小外接矩形
    rect = cv2.minAreaRect(reg_box)
    # 获取矩形的四个角点
    box = cv2.boxPoints(rect)
    box = np.int0(box).tolist()

    reg_box = sort_reg_box(box)

    FOV_SHAPE_x_y, FOV_PIXES_x_y = get_fov_info(pyramid_img_path)

    print("End cut pic, start stitch pic...")

    save_dir = pyramid_img_path.replace(".ome.tif", "")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    stitch_img = new_stitch(pyramid_img_path, reg_box, FOV_PIXES_x_y=FOV_PIXES_x_y, save_dir=save_dir)

    return stitch_img


def draw_img_by_json(stitch_json_path, save_dir, page=0):
    stitch_json = load_json(stitch_json_path)
    M = stitch_json["M"]
    rot_angle = stitch_json["angle"]
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]

    whole_img = pyvips.Image.new_from_file(stitch_json["pyramid_img_path"], page=page)
    # stitch_img = np.zeros((*conf.whole_img_size[::-1], 3), dtype=np.uint8)
    stitch_img = BmTiff.draw_black(*conf.whole_img_size)

    if whole_img.max() > 0:  #全黑图像，直接返回一个黑图，减少无效计算
        print("start match img---")
        x_y_range, all_index_x_y = stitch_json["x_y_range"], stitch_json["all_index_x_y"]
        for index_x_y in all_index_x_y:
            index_x, index_y = index_x_y
            if "ori_{}_{}.tif".format(index_y, index_x) not in stitch_json:
                print("ori_{}_{}.tif not have loc, skip it...".format(index_y, index_x))
                continue
            real_loc = stitch_json["ori_{}_{}.tif".format(index_y, index_x)][0]
            print("ori_{}_{}.tif : {}".format(index_y, index_x, real_loc))

            # img_ori = cv2.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))
            crop_range = (index_x * FOV_PIXES_x_y[0], index_y * FOV_PIXES_x_y[1],
                          FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
            drop_pixes = 3  # 丢掉边缘像素，避免黑线
            crop_range = tuple(crop_range + np.array((drop_pixes, drop_pixes, -2*drop_pixes, -2*drop_pixes)))
            img_ori = whole_img.crop(*crop_range)

            # real_loc_crop = (real_loc[0], real_loc[1], FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
            # sub_stitch = stitch_img.crop(*real_loc_crop)
            #
            # img_merge = merge_two_img(img_ori, sub_stitch)

            stitch_img = stitch_img.draw_image(img_ori, real_loc[0] + drop_pixes, real_loc[1] + drop_pixes)
            # show_img(stitch_img.crop(0, 0, 5000, 5000).numpy())
            pass
    print("start save {}".format(os.path.join(save_dir, r"new_stitch_img.tif")))

    # 旋转正方向
    stitch_img_crop = BmTiff.rotate_and_cinter_crop(stitch_img, rot_angle)
    # BmTiff.save_pyramid_tif(stitch_img_crop, os.path.join(save_dir, "stitch_img.ome.tif"))
    #
    std_circle_reg_box = stitch_json["std_reg_box"]
    stitch_img_crop = stitch_img_crop.crop(std_circle_reg_box[0][0], std_circle_reg_box[0][1],
                                           std_circle_reg_box[1][0]-std_circle_reg_box[0][0],
                                           std_circle_reg_box[1][1]-std_circle_reg_box[0][1])

    # BmTiff.save_pyramid_tif(stitch_img_crop, os.path.join(save_dir, "img_dist.ome.tif"))


    # 画底板
    # from need.CorrectWholeImg import cal_zoom_rate, gen_std_board_loc, gen_std_board_img
    # zoom_scale = cal_zoom_rate(img_dist.shape[1], img_dist.shape[0])
    # std_kp_loc, std_w_h = gen_std_board_loc(zoom_scale)
    # img_dist_with_board = gen_std_board_img(img_dist.shape[1], img_dist.shape[0], std_kp_loc, save_dir,
    #                                         base_img=img_dist, mask_color=conf.std_mask_color)
    # tifffile.imwrite(os.path.join(save_dir, "img_dist_with_board.tif"), img_dist_with_board,
    #                  compression=conf.compression_mode)
    ##

    return stitch_img, stitch_img_crop


def draw_img_by_json_numpy(stitch_json_path, save_dir, page=0):
    stitch_json = load_json(stitch_json_path)
    M = stitch_json["M"]
    rot_angle = stitch_json["angle"]
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]

    whole_img = tifffile.imread(stitch_json["pyramid_img_path"])
    whole_img = np.transpose(whole_img, (1, 2, 0))  # 变换为RGB
    stitch_img = np.zeros((*conf.whole_img_size[::-1], 3), dtype=np.uint8)
    # stitch_img = BmTiff.draw_black(*conf.whole_img_size)

    if True:
        print("start match img---")
        x_y_range, all_index_x_y = stitch_json["x_y_range"], stitch_json["all_index_x_y"]
        for index_x_y in all_index_x_y:
            index_x, index_y = index_x_y
            if "ori_{}_{}.tif".format(index_y, index_x) not in stitch_json:
                print("ori_{}_{}.tif not have loc, skip it...".format(index_y, index_x))
                continue
            real_loc = stitch_json["ori_{}_{}.tif".format(index_y, index_x)][0]
            print("ori_{}_{}.tif : {}".format(index_y, index_x, real_loc))

            # img_ori = cv2.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))
            crop_range = (index_x * FOV_PIXES_x_y[0], index_y * FOV_PIXES_x_y[1],
                          FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
            drop_pixes = 3  # 丢掉边缘像素，避免黑线
            crop_range = tuple(crop_range + np.array((drop_pixes, drop_pixes, -2*drop_pixes, -2*drop_pixes)))
            img_ori = whole_img[crop_range[1]: crop_range[1] + crop_range[3],
                                crop_range[0]: crop_range[0] + crop_range[2], ...]

            # real_loc_crop = (real_loc[0], real_loc[1], FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
            # sub_stitch = stitch_img.crop(*real_loc_crop)
            #
            # img_merge = merge_two_img(img_ori, sub_stitch)
            real_loc_drop = (real_loc[0] + drop_pixes, real_loc[1] + drop_pixes)
            stitch_img[real_loc_drop[1]:real_loc_drop[1]+img_ori.shape[0],
                       real_loc_drop[0]:real_loc_drop[0]+img_ori.shape[1], ...] = img_ori
            # show_img(stitch_img.crop(0, 0, 5000, 5000).numpy())
            pass
    print("start save {}".format(os.path.join(save_dir, r"new_stitch_img.tif")))

    stitch_img = pyvips.Image.new_from_array(stitch_img)
    # 旋转正方向
    stitch_img_crop = BmTiff.rotate_and_cinter_crop(stitch_img, rot_angle)
    # BmTiff.save_pyramid_tif(stitch_img_crop, os.path.join(save_dir, "stitch_img.ome.tif"))
    #
    std_circle_reg_box = stitch_json["std_reg_box"]
    stitch_img_crop = stitch_img_crop.crop(std_circle_reg_box[0][0], std_circle_reg_box[0][1],
                                           std_circle_reg_box[1][0]-std_circle_reg_box[0][0],
                                           std_circle_reg_box[1][1]-std_circle_reg_box[0][1])

    # BmTiff.save_pyramid_tif(stitch_img_crop, os.path.join(save_dir, "img_dist.ome.tif"))


    # 画底板
    # from need.CorrectWholeImg import cal_zoom_rate, gen_std_board_loc, gen_std_board_img
    # zoom_scale = cal_zoom_rate(img_dist.shape[1], img_dist.shape[0])
    # std_kp_loc, std_w_h = gen_std_board_loc(zoom_scale)
    # img_dist_with_board = gen_std_board_img(img_dist.shape[1], img_dist.shape[0], std_kp_loc, save_dir,
    #                                         base_img=img_dist, mask_color=conf.std_mask_color)
    # tifffile.imwrite(os.path.join(save_dir, "img_dist_with_board.tif"), img_dist_with_board,
    #                  compression=conf.compression_mode)
    ##

    return stitch_img, stitch_img_crop


def find_distance(img, rot=None, micro_rate=int(conf.conf.get("match-imgs", "auto_std_d_sub_rate"))):
    auto_std_d_min = int(conf.conf.get("match-imgs", "auto_std_d_min"))//2
    auto_std_d_max = int(conf.conf.get("match-imgs", "auto_std_d_max"))//2

    # if conf.base_mode == "S2000-2":  # S2000-2减小尺寸做计算
    #     auto_std_d_min = auto_std_d_min//2
    #     auto_std_d_max = auto_std_d_max//2
    # print(micro_rate, auto_std_d_min, auto_std_d_max)
    # 利用寻峰方法找到一张二值化图的平均分界线距离
    if rot is not None:  # 旋转角度至水平
        img = BmTiff.rotate_and_cinter_crop(img, rot)

    # show_img(img.numpy())
    img = img.numpy()
    x_sum = np.sum(img, axis=0, dtype=int)

    # x_sum = np.interp(np.linspace(0, len(x_sum)-1, 10000), np.arange(len(x_sum)), x_sum)

    fft_signal = abs(np.fft.fft(x_sum, n=len(x_sum)*micro_rate))
    # 控制数据范围，避免离谱数据
    fft_signal[0] = 0
    fft_signal[:len(fft_signal)//auto_std_d_max] = 0
    fft_signal[len(fft_signal)//auto_std_d_min:] = 0

    # import matplotlib.pyplot as plt
    # plt.plot(fft_signal)
    # plt.plot(x_sum)
    f = np.argmax(fft_signal) / micro_rate

    # show_img(img)
    # plt.close()
    return len(x_sum)/f * 2  # x方向统计纵轴频率间距，由于微球是错行相隔，所以频率会翻倍，所以计算“周期”要*2


def calculate_std_d(stitch_json):
    pyramid_img_path = stitch_json["pyramid_img_path"]
    whole_img = BmTiff.read_pyramid_from_file(pyramid_img_path)
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]
    angle = stitch_json["angle"]
    x_y_range = stitch_json["x_y_range"]

    all_mean_distance = []
    for i in range(20):
        # 随机选择序号，避开边缘
        random_x = random.randint(x_y_range[0][0] + 1, x_y_range[1][0] - 1)
        random_y = random.randint(x_y_range[0][1] + 1, x_y_range[1][1] - 1)
        crop_range = (random_x * FOV_PIXES_x_y[0], random_y * FOV_PIXES_x_y[1],
                      FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
        img = whole_img.crop(*crop_range)
        mean_distance = find_distance(img, rot=angle)
        all_mean_distance.append(mean_distance)
        print(f"crop: {crop_range} -> mean_distance:{mean_distance}")
    return np.median(all_mean_distance)


def find_error_pic(error_point, stitch_json):
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]
    # 遍历cycle_json中每一个视野的位置，定位error位置的视野序号
    error_point_x, error_point_y = error_point
    for index_x_y in stitch_json["all_index_x_y"]:
        img_name = f"ori_{index_x_y[1]}_{index_x_y[0]}.tif"
        left_top_loc = stitch_json[img_name][0]
        if left_top_loc[0] <= error_point_x <= left_top_loc[0] + FOV_PIXES_x_y[0] and \
                left_top_loc[1] <= error_point_y <= left_top_loc[1] + FOV_PIXES_x_y[1]:
            return img_name, left_top_loc, index_x_y
    raise Exception("Error: Can't find error pic by error point:{}. Check it please!".format(error_point))


def draw_part_circle_pic(circle_centers, stitch_json, start_loc, shape=(conf.barcode_size_x, conf.barcode_size_y)):
    std_d = stitch_json["std_circles_d"]
    std_r = int(std_d * 0.4)

    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]
    start_loc = np.asarray(start_loc)
    end_loc = start_loc + FOV_PIXES_x_y
    # 找到开始位置和结束位置最接近的圆心坐标
    distance_values = np.sum(np.abs(circle_centers - start_loc), axis=2)
    x_i_start, y_i_start = cv2.minMaxLoc(distance_values)[2][::-1]

    distance_values = np.sum(np.abs(circle_centers - end_loc), axis=2)
    x_i_end, y_i_end = cv2.minMaxLoc(distance_values)[2][::-1]

    circle_img = np.zeros((*FOV_PIXES_x_y[::-1], 3), dtype=np.uint8)
    for x in range(x_i_start, x_i_end+1):
        for y in range(y_i_start, y_i_end+1):
            if y % (shape[1] + 1) == 0 or x % (shape[0] + 1) == 0:
                # 不同块的分割线
                continue
            offset_loc = circle_centers[x, y] - start_loc
            cv2.circle(circle_img, tuple(offset_loc), std_r, (1, 155, 1), 1)
    return circle_img


def correct_img(wrong_point_norm, stitch_json_path):
    stitch_json = load_json(stitch_json_path)
    whole_img = BmTiff.read_pyramid_from_file(stitch_json["pyramid_img_path"], page=conf.stitch_channal)
    FOV_PIXES_x_y = stitch_json["FOV_PIXES_x_y"]
    std_reg_box = stitch_json["std_reg_box"]
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
    error_img_name, error_img_loc, index_x_y = find_error_pic(real_point, stitch_json)
    crop_range = (index_x_y[0] * FOV_PIXES_x_y[0], index_x_y[1] * FOV_PIXES_x_y[1],
                  FOV_PIXES_x_y[0], FOV_PIXES_x_y[1])
    error_img = whole_img.crop(*crop_range).numpy()
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
    pyramid_img_path = r"C:\Users\ofen\Documents\WXWork\1688855473391904\Cache\File\2025-04\result.ome.tif"
    # reg_box = [[1658, 2248], [14113, 2222], [14136, 14689], [1693, 14713]]
    # # pyramid_img_path = r"Z:\work\tmp\merge.ome.tif"
    # # pyramid_img_path = r"/share/nas1/ouyangfeng/work/tmp/merge.ome.tif"
    # # reg_box = [[1196, 5529], [91849, 5501], [91900, 112668], [1246, 112690]]
    # # whole_img = pyvips.Image.new_from_file(pyramid_img_path)
    #
    # stitch_img = cut_and_stitch(pyramid_img_path, reg_box)

    # img = BmTiff.read_pyramid_from_file(r"Z:\work\tmp\50_60_decode\10X 5060细胞分割11.12\DG31EN03F2\HE\IMG002x007.tif")
    # img = BmTiff.read_pyramid_from_file(r"Z:\work\tmp\DI29FN01F1\IMG015x007.tif")
    # img = img[0].rot270()
    # a = find_distance(img)
    # 校正错误示例
    # stitch_json_path = r"E:\test\A_big_tiff\20Xmerge\stitch_json.json"
    # wrong_point = (7528, 10838)
    # wrong_point_norm = (7528/24770, 10838/24911)  # 为了避免坐标位置尺度不一致，采用归一化的坐标位置
    # correct_img(wrong_point_norm, stitch_json_path)

    # ome_tif_path = r"E:\test\tmp\small_merge\img_dist_merge.ome.tif"
    # output_tif_path = r"E:\test\tmp\small_merge\img_dist.tif"
    #
    # # 目标高度
    # out_size_rate_w_h = ((conf.barcode_size_x + 1) * 2 * conf.base_size_x) / (
    #             (conf.barcode_size_y + 1) * np.sqrt(3) * conf.base_size_y)
    # # img_dist = cv2.resize(img_dist, (int(conf.out_size * out_size_rate_w_h), conf.out_size))
    # new_size = (int(conf.out_size * out_size_rate_w_h), conf.out_size)
    #
    # BmTiff.save_special_size(ome_tif_path, output_tif_path, new_size[0], new_size[1], compression="lzw")
