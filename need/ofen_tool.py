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
    if max_len > 32768:
        # from skimage import transform
        # warped_image = transform.warp(src, np.linalg.inv(M), output_shape=dsize[::-1])
        # return (warped_image * 255).astype(np.uint8)

        scale_rate = 32000 / max_len
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
