import copy
import json
import time
import os
import time
import sys
import errno
import random
import cv2
import numpy as np
import pyautogui
from retry import retry


class FileLockException(Exception):
    pass


class FileLock(object):

    def __init__(self, file_name, timeout=10, delay=.05):

        self.is_locked = False
        # 将锁文件放置统一位置，方便管理
        dirs = "./lock"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        self.lockfile = os.path.join(dirs, "%s.lock" % file_name)
        self.file_name = file_name
        self.timeout = timeout
        self.delay = delay

    def acquire(self):
        start_time = time.time()
        while True:
            try:
                # 独占式打开文件
                # os.O_RDWR : 以读写的方式打开
                # os.O_CREAT: 创建并打开一个新文件
                # os.O_EXCL: 如果指定的文件存在，返回错误
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if (time.time() - start_time) >= self.timeout:
                    raise FileLockException("Timeout occured.")
                time.sleep(self.delay)
        self.is_locked = True

    def release(self):
        # 关闭文件，删除文件
        if self.is_locked:
            os.close(self.fd)
            os.unlink(self.lockfile)
            self.is_locked = False

    def __enter__(self):
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        if self.is_locked:
            self.release()

    def __del__(self):
        self.release()


def time_cost(fn):
    """
    统计耗时装饰器
    :param fn: 待装饰函数
    :return:被装饰的函数
    """
    def warp(*args, **kwargs):
        t1 = time.time()
        res = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: %s use %s" % (fn.__name__, t2 - t1))
        return res
    return warp


@retry(tries=10)
def load_json(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data


@retry(tries=10)
def save_json(json_path, data):
    with FileLock(json_path, timeout=8):
        with open(json_path, "w") as fp:
            json.dumps(data)  # 先格式化，避免dump报错破坏原始文件
            json.dump(data, fp, indent=4)


def show_img(pic, name=None, line_width=3):
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, pic[y, x])
        if event == cv2.EVENT_RBUTTONDOWN:
            param[:] = [x, y]
            cv2.waitKey(300)
            cv2.destroyWindow(name)
        if event == cv2.EVENT_MOUSEMOVE:
            n_pic = cv2.line(copy.copy(pic), (x, 0), (x, pic.shape[0]), (0, 255, 0), line_width)
            n_pic = cv2.line(n_pic, (0, y), (pic.shape[1], y), (0, 255, 0), line_width)
            cv2.imshow(name, n_pic)

    if not name:
        name = str(random.random())
    cv2.namedWindow(name, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, (800, 600))
    cv2.imshow(name, pic)
    tmp = []
    cv2.setMouseCallback(name, mouse, tmp)
    cv2.waitKey()
    try:
        cv2.destroyWindow(name)
    except:
        pass
    return tmp[::-1]


def my_conv2d(img, conv_mask, start_loc=(0, 0), end_loc=None,
              step=1, scan_range=None, rough_adj=False, negative_mask=None):
    '''
    # 注意，输入均为二值化图像，名叫卷积，实则只是统计每次掩模选出的为255的像素数目
    :param img: 输入二值化图像
    :param conv_mask: 输入模板
    :param start_loc: 开始位置，默认从（0，0）
    :param end_loc: 结束位置，默认计算图像和目标长宽之差
    :param step: 步长，若不为1，则先按照step粗调，选出最匹配位置，再周围细调
    :param scan_range: 扫描范围，与图像中心点的差值，如（20，20），表示匹配扫描只在中心点位置上下左右20像素范围匹配
            ，该参数与开始位置和结束位置冲突，如填写，则start_loc与end_loc失效
    :param rough_adj: 粗调，当为true，不再对步长之内做细调
    :param negative_mask: 负模板，shape与img相同，当非空，除了会对原图计算相关值，还会将conv_mask与此负模板也做卷积相关，相关值作为负数减去
    :return: 匹配矩阵
    '''

    if end_loc is None:
        end_loc = np.asarray(img.shape) - np.asarray(conv_mask.shape) + 1

    if scan_range:
        start_loc = (np.asarray(img.shape) - np.asarray(conv_mask.shape)) // 2 - scan_range
        start_loc = list(map(lambda x: max(x, 0), start_loc))
        end_loc = (np.asarray(img.shape) - np.asarray(conv_mask.shape)) // 2 + scan_range
        max_end_loc = np.asarray(img.shape) - np.asarray(conv_mask.shape) + 1
        end_loc = [min(max_end_loc[0], end_loc[0]), min(max_end_loc[1], end_loc[1])]
    # print(start_loc)

    x, y = conv_mask.shape
    result = np.zeros((img.shape - np.asarray([x - 1, y - 1]))[::-1], dtype=int)  # shape反向主要是卷积的转置才是相关结果
    for i in range(start_loc[0], end_loc[0], step):
        for j in range(start_loc[1], end_loc[1], step):
            if i < 0 or j < 0:
                continue
            result[j, i] = cv2.countNonZero(img[i: i + x, j: j + y] & conv_mask)
            if negative_mask is not None:
                negative_result = cv2.countNonZero(negative_mask[i: i + x, j: j + y] & conv_mask)
                result[j, i] -= 10 * negative_result
    if not step == 1 and not rough_adj:
        step = step // 2
        max_x, max_y = cv2.minMaxLoc(result)[3]
        for i in range(max(max_x - step, 0), min(max_x + step, result.shape[0])):
            for j in range(max(max_y - step, 0), min(max_y + step, result.shape[1])):
                result[j, i] = cv2.countNonZero(img[i: i + x, j: j + y] & conv_mask)
                # if negative_mask is not None:
                #     negative_result = cv2.countNonZero(negative_mask[i: i + x, j: j + y] & conv_mask)
                #     result[j, i] -= 3*negative_result
    return result


def my_warpPerspective(src, M, dsize, **kwargs):
    # 对于尺寸超出限制的图像，采取先缩放，再映射，再放大回原尺寸
    max_len = max(src.shape)
    if max_len > 32767:
        # from skimage import transform
        # warped_image = transform.warp(src, np.linalg.inv(M), output_shape=dsize[::-1])
        # return (warped_image * 255).astype(np.uint8)

        scale_rate = 30000 / max_len
        M[0][2] = M[0][2] * scale_rate
        M[1][2] = M[1][2] * scale_rate
        scale_img = cv2.resize(src, (int(src.shape[1] * scale_rate), int(src.shape[0] * scale_rate)))
        scale_img = cv2.warpPerspective(scale_img, M, scale_img.shape[:2][::-1], **kwargs)
        return cv2.resize(scale_img, src.shape[:2][::-1])
    else:
        return cv2.warpPerspective(src, M, dsize, **kwargs)
    pass


def gamma_trans(img, gamma):
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)


def shift_move_show_img(fixed_pic, var_pic, name=None, scale_size=None, line_width=3, center_point=None):
    # 取图像中心部分区域对齐，减少计算量，提高流畅度
    if scale_size:
        if center_point is None:
            height, width = fixed_pic.shape[:2]
            start_x, start_y = int((width - scale_size[0]) / 2), int((height - scale_size[1]) / 2)
        else:
            height, width = fixed_pic.shape[:2]
            # 先对终止边界处理
            center_point[0] = width - scale_size[0]//2 if center_point[0] + scale_size[0] > width else center_point[0]
            center_point[1] = height - scale_size[1]//2 if center_point[1] + scale_size[1] > height else center_point[1]
            # 先对起始界处理
            start_x = center_point[0] - scale_size[0]//2 if center_point[0] - scale_size[0]//2 > 0 else 0
            start_y = center_point[1] - scale_size[1]//2 if center_point[1] - scale_size[1]//2 > 0 else 0
        end_x, end_y = start_x + scale_size[0], start_y + scale_size[1]
        fixed_pic = fixed_pic[start_y:end_y, start_x:end_x]
        var_pic = var_pic[start_y:end_y, start_x:end_x]

    shift = np.array([0, 0])
    l_press = False
    move_rate = 10  # 移动减速比，图像移动缩减比率
    tmp_start_pos = None

    def mouse(event, x, y, flags, param):
        nonlocal shift
        nonlocal l_press
        nonlocal move_rate
        nonlocal tmp_start_pos

        if event == cv2.EVENT_LBUTTONDOWN:
            l_press = True
            tmp_start_pos = [x, y]
            print("start:", x, y, pic[y, x])
        if event == cv2.EVENT_LBUTTONUP and l_press:
            if tmp_start_pos:
                l_press = False
                tmp_shift = (np.array([x, y]) - tmp_start_pos)//move_rate
                shift += tmp_shift
                print(f"shift:{shift}")
                tmp_start_pos = [0, 0]
                # print("end:", x, y, pic[y, x])
                # shift += 1
        if event == cv2.EVENT_RBUTTONDOWN:
            param[:] = shift[:]
            cv2.waitKey(300)
            pyautogui.press("enter")
            # cv2.destroyWindow(name)
        if event == cv2.EVENT_MOUSEMOVE:
            if l_press:  # 左键按下且移动
                tmp_loc_x = shift[0] + (x - tmp_start_pos[0])//move_rate
                tmp_loc_y = shift[1] + (y - tmp_start_pos[1])//move_rate

                # 平移图像
                shift_M = np.float32([[1, 0, tmp_loc_x], [0, 1, tmp_loc_y]])
                translated_image = cv2.warpAffine(var_pic, shift_M, (var_pic.shape[1], var_pic.shape[0]))
                # 叠加显示
                translated_image = np.where(fixed_pic > 0, fixed_pic, translated_image)
                # x, y = tmp_loc_x, tmp_loc_y
                # n_pic = cv2.line(copy.copy(pic), (x, 0), (x, pic.shape[0]), (0, 255, 0), line_width)
                # n_pic = cv2.line(n_pic, (0, y), (pic.shape[1], y), (0, 255, 0), line_width)
                cv2.imshow(name, translated_image)

    if not name:
        name = str(random.random())
    cv2.namedWindow(name, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(name, 800, 600)
    pic = np.where(fixed_pic > 0, fixed_pic, var_pic)
    cv2.imshow(name, pic)
    tmp = []
    cv2.setMouseCallback(name, mouse, tmp)
    # cv2.waitKey()
    # 避免点X后死锁
    while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
        # print(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE))
        k = cv2.waitKey(100)
        if k != -1:
            break
    try:
        cv2.destroyWindow(name)
    except:
        pass
    return tmp


def gray_enhance(gray, black=2, white=70, gamma=1.69):
    gray = np.where(gray < black, black, gray)
    gray = np.where(gray > white, white, gray)
    gray = np.power((gray - black) / (white - black), 1.0 / gamma)
    gray = np.round(gray * 255.0)
    gray = np.array(gray, dtype='uint8')
    return gray