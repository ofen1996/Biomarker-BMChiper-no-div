import os
import time

import cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from need.ofen_tool import show_img


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

    plt.plot(angles, grad_values, marker='o')
    best_angle = angles[np.argmax(grad_values)]
    best_rot_matrix = rot_matrixs[np.argmax(grad_values)]

    # print(f"Best angle:{best_angle}")

    # rotated_image = cv2.warpPerspective(img, best_rot_matrix, img.shape[:2][::-1])
    # show_img(rotated_image)

    return best_angle, best_rot_matrix


def find_distance(img, M=None, conv_len=20, peaks_threshold=0.25, peaks_min_distance=200):
    # 利用寻峰方法找到一张二值化图的平均分界线距离
    if M is not None:
        img = cv2.warpPerspective(img, M, img.shape[:2][::-1])

    img_b = img[..., 0] | img[..., 1] | img[..., 2]
    # img_b = img[..., 0]  # 只用单通道效果更好
    # show_img(img_b)
    img_b = cv2.bitwise_not(img_b)
    x_sum = np.sum(img_b, axis=0, dtype=int)
    fft_signal = abs(np.fft.fft(x_sum))
    # 控制数据范围，避免离谱数据
    fft_signal[0] = 0
    fft_signal[:len(x_sum)//10] = 0
    fft_signal[len(x_sum)//5:] = 0    # 只计算10-20之间的结果

    # import matplotlib.pyplot as plt
    # plt.plot(fft_signal)
    # plt.plot(x_sum)
    f = np.argmax(fft_signal)
    return len(x_sum) / f * 2


# 示例使用
if __name__ == "__main__":
    # 读取一张图像
    img = cv2.imread(r"C:\Users\ofen\Documents\WXWork\1688855473391904\Cache\File\2024-05\ori_2_6.tif")
    t1 = time.time()
    best_angle, best_rot_matrix = find_best_rotate(img)
    print(f"time cost: {time.time()-t1}")
    std_d = find_distance(img, M=best_rot_matrix)
