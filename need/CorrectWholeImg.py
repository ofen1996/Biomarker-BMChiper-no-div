# import time
import os
import cv2
import tifffile
import numpy as np
import configparser
# import skimage
# import tifffile
#
# from need.ofen_tool import show_img
from need.KpDetectByYolo import MyDetector
from need.config import conf


h_num, w_num = (10, 10)  # 切成100份
try:
    detector = MyDetector("./model/best.onnx")
except:
    detector = MyDetector("../model/best.onnx")


# 依据HE图片的实际大小进行缩放比例计算
# width：HE图片的宽
# height: HE图片的高
def cal_zoom_rate(width, height):
    std_width = 1000
    std_height = std_width / (conf.base_size_x * (conf.barcode_size_x+1)) * (conf.base_size_y * (conf.barcode_size_y+1) * np.sqrt(3) / 2.0)
    if std_width / std_height > width / height:
        scale = width / std_width
    else:
        scale = height / std_height
    return scale


def gen_std_board_loc(zoom_scale):
    img_width = 1000
    std_kp_loc = np.zeros((conf.base_size_x - 1, conf.base_size_y - 1, 2), dtype=float)
    std_w = 1.0 * img_width / conf.base_size_x / (conf.barcode_size_x+1) * zoom_scale
    std_h = std_w * np.sqrt(3) / 2

    for h in range(conf.base_size_y - 1):
        for w in range(conf.base_size_x - 1):
            h_loc = (h + 1) * (conf.barcode_size_y+1) * std_h
            w_loc = (w + 1) * (conf.barcode_size_x+1) * std_w - std_w / 2  # 一定是偶数行
            std_kp_loc[w, h] = w_loc, h_loc
    return std_kp_loc, (std_w, std_h)


def gen_std_board_img(width, height, std_kp_loc=None, save_dir=None, base_img=None, mask_color=(255, 255, 255)):
    zoom_scale = cal_zoom_rate(width, height)
    img_width = 1000
    std_w = 1.0 * img_width / conf.base_size_x / (conf.barcode_size_x+1) * zoom_scale
    std_h = std_w * 0.5 * 3 ** 0.5

    radius = round(std_w * 0.618 / 2) if round(std_w * 0.618 / 2) > 1 else 1
    if base_img is None:
        std_board = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        std_board = base_img
    for h in range(conf.base_size_y * (conf.barcode_size_y+1)):
        for w in range(conf.base_size_x * (conf.barcode_size_x+1)):
            if h % (conf.barcode_size_y+1) == 0 or w % (conf.barcode_size_x+1) == 0:
                continue  # 边界点跳过
            tmp_w = w * std_w
            tmp_h = h * std_h
            if h % 2 == 0:  # 偶数行，w位置减去1/2
                tmp_w = tmp_w - std_w * 0.5
            cv2.circle(std_board, (round(tmp_w), round(tmp_h)), radius, mask_color, -1)

    if std_kp_loc is not None:
        # 画一个标准点位参考图
        for y in range(conf.base_size_y - 1):
            for x in range(conf.base_size_x - 1):
                cv2.circle(std_board, tuple(map(int, std_kp_loc[x, y])), 11, 255, -1)
        if conf.DEBUG:
            cv2.imwrite(os.path.join(save_dir, "std_kp_img.tif"), std_board)

    return std_board


def locate_split(div_sum, base_num):
    # 通过双指针方法，找到归一化数据div_sum的base_num个分割区域
    split = []
    tmp_start = 0
    tmp_end = 0
    for i, y_sum in enumerate(div_sum):
        if i < tmp_end:  # 双指针，i小于下降沿跳过
            continue
        if y_sum >= 0.2 and tmp_end >= tmp_start:
            # 上升沿
            # print("up", i)
            tmp_start = i
            tmp_end = tmp_start + 1
            for j in range(tmp_end, tmp_end + (len(div_sum) // 50)):
                if div_sum[j] >= 0.2:
                    continue
                else:
                    tmp_end = j
                    break
            split.append([tmp_start, tmp_end])

    # 为了防止某行为空，导致序号跳变，需要根据梯度中位数重新序列化
    new_split = [[0, 0] for x in range(base_num)]
    distance = float(np.median(np.gradient(list(map(np.mean, split)))))
    for sub_range in split:
        mean_sub_range = np.mean(sub_range)
        index = round(mean_sub_range / distance) - 1
        new_split[index] = sub_range

    return new_split


def detect_kp(img, detect_channel=None):
    # 预测全图的所有关键点。将原图裁剪为h_num, w_num后，使用yolo预测关键点
    size_h, size_w = img.shape[:2]
    distance_h, distance_w = size_h // h_num, size_w // w_num

    k_p_loc = []
    for div_h in range(h_num):
        print(div_h)
        for div_w in range(w_num):
            start_h, start_w = div_h * distance_h, div_w * distance_w
            # 截取图像外扩0.05*distance，避免边缘识别效果不好
            distance_add = int(min(distance_h, distance_w) * 0.05)
            part_img = img[max(0, start_h - distance_add): start_h + int(distance_h * 1.05),
                           max(0, start_w - distance_add): start_w + int(distance_w * 1.05), :]
            if detect_channel is not None:  # 给定识别通道
                out_img, centers = detector.detect(cv2.cvtColor(part_img[..., detect_channel], cv2.COLOR_GRAY2BGR), conf.kp_detect_confidence)
            else:
                out_img, centers = detector.detect(part_img, conf.kp_detect_confidence)
            for center in centers:
                center[:2] = center[:2] + np.asarray((max(0, start_w - distance_add), max(0, start_h - distance_add)))
            # 过滤掉边缘的识别点，避免重复
            centers = [center for center in centers if (start_w < center[0] < start_w + distance_w and
                                                        start_h < center[1] < start_h + distance_h)]
            k_p_loc.extend(centers)
            # show_img(out_img)
            # show_img(part_img)
            pass
    print("End detect kp")
    return k_p_loc


def filter_kp(k_p_loc, judge_range, img, save_dir="./"):

    kp_range = judge_range // 2
    print(judge_range)
    # 绘制关键点的重合区域，以此作为众数过滤偏移过大的错误点
    label = np.zeros(img.shape[:2], dtype=np.uint8)
    for kp in k_p_loc:
        x, y, tmp_conf = kp
        if not (judge_range * 2 < x < img.shape[1] - judge_range * 2 and
                judge_range * 2 < y < img.shape[0] - judge_range * 2):
            continue  # skip edge point
        # cv2.circle(label, (x, y), 3, (255, 255, 255), -1)
        label[:, x - kp_range: x + kp_range] += 255 // conf.base_size_x // 2
        label[y - kp_range: y + kp_range, :] += 255 // conf.base_size_y // 2
        # cv2.line(label, (x, 0), (x, img.shape[0]), (255//46), judge_range)
        # cv2.line(label, (0, y), (img.shape[1], y), (255//46), judge_range)
    if conf.DEBUG:
        cv2.imwrite(os.path.join(save_dir, "./label.tif"), label)
    print("End draw label")

    # 根据label筛选可信区域，过滤k_p
    label_nom = label / label.max()  # 归一化
    kp_loc_confidence = np.where(label_nom > conf.kp_loc_confidence, 1.0, 0)  # 置信区域
    kp_loc_confidence = cv2.dilate(kp_loc_confidence, kernel=np.ones(shape=[3, 3]))
    kp_loc_confidence = cv2.erode(kp_loc_confidence, kernel=np.ones(shape=[3, 3]))
    wrong_kp = []
    kp_final = []
    for i, kp in enumerate(k_p_loc):
        if not kp_loc_confidence[kp[1], kp[0]]:
            wrong_kp.append(kp)
            cv2.circle(img, (kp[0], kp[1]), 13, (0, 0, 255), -1)
        else:
            kp_final.append(kp)
            # 在图像上画点
            cv2.circle(img, (kp[0], kp[1]), 3, (0, 0, 0), -1)
            cv2.putText(img, str(round(kp[2], 2)), (int(kp[0]), int(kp[1])), 0, 1, (0, 255, 0),
                        thickness=2)

    print("End draw img")
    if conf.DEBUG:
        cv2.imwrite(os.path.join(save_dir, "./kp_filter.tif"), img)
    return kp_final, wrong_kp, kp_loc_confidence


def kp_serialize(kp_final, kp_loc_confidence, judge_range):
    # 将关键点按顺序定位，序列化
    y_div_sum = [sum(kp_loc_confidence[i, :]) for i in range(kp_loc_confidence.shape[0])]
    x_div_sum = [sum(kp_loc_confidence[:, i]) for i in range(kp_loc_confidence.shape[1])]
    y_div_sum = np.convolve(y_div_sum, np.ones(5), 1)  # 抹平每一行的波动
    x_div_sum = np.convolve(x_div_sum, np.ones(5), 1)  # 抹平每一行的波动
    y_div_sum = y_div_sum / y_div_sum.max()  # 归一化
    x_div_sum = x_div_sum / x_div_sum.max()  # 归一化
    y_range_split = locate_split(y_div_sum, conf.base_size_y)
    x_range_split = locate_split(x_div_sum, conf.base_size_y)  # [[99, 136], [217, 255], ...] 统计每个点位出现的范围 len=45

    # 将所有kp定位回(45, 45)位置矩阵内
    real_kp_loc = np.zeros((conf.base_size_x, conf.base_size_y, 3), dtype=float)
    for kp in kp_final:
        x, y, tmp_conf = kp
        # print(kp)
        x_index = abs(np.asarray(x_range_split) - x).argmin() // 2  # 找到距离x_range最近的索引，//2是因为 argmin是展平检索最小值
        y_index = abs(np.asarray(y_range_split) - y).argmin() // 2
        rel_loc = (sum(x_range_split[x_index]) / 2, sum(y_range_split[y_index]) / 2)  # 参考位置
        if np.linalg.norm(rel_loc - np.asarray([x, y])) > judge_range * np.sqrt(2):  # 如果欧式距离差距过大，则抛弃之
            continue
        if real_kp_loc[x_index, y_index][2] < tmp_conf:  # 如果一个索引内有 多个点，取置信度最高的
            real_kp_loc[x_index, y_index] = kp
    return real_kp_loc


def kp_auto_complete(real_kp_loc):
    # 根据现有kp位置，求出相邻点的间距的中位数x_spacing, y_spacing
    x_spacing, y_spacing = (np.median(np.gradient(real_kp_loc[:, :, 0])[0]),
                            np.median(np.gradient(real_kp_loc[:, :, 1])[1]))

    # 补全缺失点
    real_kp_loc_plus = real_kp_loc.copy()
    for y in range(conf.base_size_y):
        for x in range(conf.base_size_x):
            x_loc, y_loc, tmp_conf = real_kp_loc_plus[x, y]
            if tmp_conf == 0:
                # 从上下左右4个点找1个点计算
                if y - 1 >= 0 and real_kp_loc_plus[x, y - 1][2] > 0:
                    x_loc = real_kp_loc_plus[x, y - 1][0]
                    y_loc = real_kp_loc_plus[x, y - 1][1] + y_spacing

                elif y + 1 < conf.base_size_y - 1 and real_kp_loc_plus[x, y + 1][2] > 0:
                    x_loc = real_kp_loc_plus[x, y + 1][0]
                    y_loc = real_kp_loc_plus[x, y + 1][1] - y_spacing

                elif x - 1 >= 0 and real_kp_loc_plus[x - 1, y][2] > 0:
                    x_loc = real_kp_loc_plus[x - 1, y][0] + x_spacing
                    y_loc = real_kp_loc_plus[x - 1, y][1]

                elif x + 1 < conf.base_size_x - 1 and real_kp_loc_plus[x + 1, y][2] > 0:
                    x_loc = real_kp_loc_plus[x + 1, y][0] - x_spacing
                    y_loc = real_kp_loc_plus[x + 1, y][1]

                real_kp_loc_plus[x, y] = [x_loc, y_loc, 0]
    return real_kp_loc_plus


def draw_kp_in_img(img, real_kp_loc_plus, save_dir=None, save_name="test_plus.tif"):
    # 画图验证点位的准确性
    for y in range(conf.base_size_y - 1):
        for x in range(conf.base_size_x - 1):
            kp = real_kp_loc_plus[x, y]
            cv2.circle(img, (int(kp[0]), int(kp[1])), 10, (0, 0, 0), 3)
            cv2.putText(img, "({},{})".format(int(x), int(y)), (int(kp[0]), int(kp[1]) + 20), 0, 1, (0, 255, 0),
                        thickness=2)
    if conf.DEBUG:
        cv2.imwrite(os.path.join(save_dir, "./{}".format(save_name)), img)
    return img


def find_homography(real_kp_loc_plus, std_kp_loc):
    if conf.use_real_kp_only:
        # 只保留识别准确的点进行单应计算
        real_kp = []
        std_kp = []
        for y in range(conf.base_size_y - 1):
            for x in range(conf.base_size_x - 1):
                x_loc, y_loc, tmp_conf = real_kp_loc_plus[x, y]
                std_x, std_y = std_kp_loc[x, y]
                if tmp_conf > conf.kp_detect_confidence:
                    real_kp.append([x_loc, y_loc])
                    std_kp.append([std_x, std_y])

        h, status = cv2.findHomography(np.asarray(real_kp), np.asarray(std_kp), 0)
    else:
        h, status = cv2.findHomography(real_kp_loc_plus.reshape(-1, 3)[:, :2], std_kp_loc.reshape(-1, 2), 0)
    return h


def correct_whole_img(img_path, detect_channel=None):
    save_dir = os.path.split(img_path)[0]
    img_name = os.path.split(img_path)[1]
    save_dir = os.path.join(save_dir, os.path.splitext(img_name)[0])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img = tifffile.imread(img_path)
    if img.shape[0] > conf.out_size:
        # out_rate = img.shape[0] / conf.out_size
        # new_size = (np.asarray(img.shape[:2]) / out_rate).astype(dtype=int)
        img = cv2.resize(img, (int(conf.out_size * 0.99435), conf.out_size))  # 芯片标准宽度比高度有个比例
    else:
        height = img.shape[0]
        weight = int(height * 0.99435)  # 芯片标准宽度比高度有个比例
        img = cv2.resize(img, (weight, height))
    img_dist = img.copy()

    zoom_rate = 1
    if img.shape[0] > conf.calculate_size:
        zoom_rate = img.shape[0] / conf.calculate_size
        new_size = (np.asarray(img.shape[:2]) / zoom_rate).astype(dtype=int)
        img = cv2.resize(img, new_size[::-1])
    # img_dist = img.copy()
    # 判断常数，以一个块的宽度0.3为标准
    judge_range = int(min(img.shape[:2]) / conf.base_size_x * 0.3)
    # 识别关键点
    k_p_loc = detect_kp(img, detect_channel=detect_channel)
    # 过滤错误点
    kp_final, wrong_kp, kp_loc_confidence = filter_kp(k_p_loc, judge_range, img, save_dir)
    # 关键点序列化，把所有关键点对应到（45， 45）的位置上
    real_kp_loc = kp_serialize(kp_final, kp_loc_confidence, judge_range)
    if conf.DEBUG:
        draw_kp_in_img(img, real_kp_loc, save_dir, save_name="tmp1.tif")  # 画一下修正后的kp位置
    # 补全缺失关键点
    real_kp_loc_plus = kp_auto_complete(real_kp_loc)

    real_kp_loc_plus = real_kp_loc_plus * zoom_rate  # 变换为实际位置
    print("Calculate Homography")
    # 画标准点图
    zoom_scale = cal_zoom_rate(img_dist.shape[1], img_dist.shape[0])
    std_kp_loc, std_w_h = gen_std_board_loc(zoom_scale)
    if conf.DEBUG:
        draw_kp_in_img(img_dist.copy(), std_kp_loc, save_dir, save_name="tmp2.tif")  # 画一下修正后的kp位置
    # 计算单应性矩阵, 对kp做std_w_h/2的位置修正，方向为左下
    real_kp_loc_plus[:, :, :2] = real_kp_loc_plus[:, :, :2] + [std_w_h[0]*conf.shift_x, std_w_h[0]*conf.shift_y]

    # 变换图像
    h = find_homography(real_kp_loc_plus, std_kp_loc)
    # # 画一个标准点位图，用作参考
    # test_dst = cv2.warpPerspective(img, h, img.shape[:2][::-1],
    #                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # std_board_img = gen_std_board_img(img.shape[1], img.shape[0], std_kp_loc, save_dir, base_img=test_dst)
    # std_board_img = cv2.cvtColor(std_board_img, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite(os.path.join(save_dir, "std_board.tif"), std_board_img)
    # test_dst = (test_dst * 0.85 + std_board_img * 0.15).astype(dtype=np.uint8)  # 叠加
    # cv2.imwrite(os.path.join(save_dir, "tmp.tif"), test_dst)

    img_dist = cv2.warpPerspective(img_dist, h, img_dist.shape[:2][::-1],
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    tifffile.imwrite(os.path.join(save_dir, "img_dist.tif"), img_dist, compression=conf.compression_mode)

    # 画标准点图
    draw_kp_in_img(img_dist, real_kp_loc_plus, save_dir)  # 画一下修正后的kp位置
    zoom_scale = cal_zoom_rate(img_dist.shape[1], img_dist.shape[0])
    std_kp_loc, std_w_h = gen_std_board_loc(zoom_scale)
    img_dist_with_board = gen_std_board_img(img_dist.shape[1], img_dist.shape[0], std_kp_loc, save_dir,
                                            base_img=img_dist, mask_color=conf.std_mask_color)
    tifffile.imwrite(os.path.join(save_dir, "img_dist_with_board.tif"), img_dist_with_board,
                     compression=conf.compression_mode)




if __name__ == '__main__':
    img_path = r"C:\Users\ofen\Documents\WXWork\1688855473391904\Cache\File\2024-02\Result-2024-02-01-133859.1_chip.tif"
    correct_whole_img(img_path, detect_channel=0)


