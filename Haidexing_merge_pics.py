import argparse
import sys
import numpy as np
from PIL import Image
import os
import tifffile
import cv2
from need.ofen_tool import show_img

## 海德星的小图拼接成大图

def generate_matrix(M, N):
    rows, cols = np.meshgrid(np.arange(1, M+1), np.arange(1, N+1), indexing='ij')
    matrix = np.stack((rows, cols), axis=-1)
    return matrix


def generate_whole_tif_RGB(pics_dir, pixes=(2048, 2448)):
    # pics_dir = r"E:\biomarker_data\test_machine\Images"
    # pixes = np.array((2048, 2448))
    img_names = os.listdir(pics_dir)

    rows = [int(name[3:6]) for name in img_names if name.endswith(".tif")]
    cols = [int(name[7:10]) for name in img_names if name.endswith(".tif")]

    # 设置矩阵形状
    row, col = max(rows), max(cols)

    # 生成矩阵
    result_matrix = generate_matrix(row, col)

    # 顺时针偏转90°
    right_index = np.rot90(result_matrix, 3)  # 存着正确的序号的矩阵
    # 打印结果
    # print(right_index)
    # 下面开始拼接图像
    # de_pixes = pixes // 2
    de_pixes = pixes

    whole_size = (de_pixes[1] * col, de_pixes[0] * row, 3)
    whole_img = np.zeros(whole_size, dtype=np.uint8)
    for r_i in range(row):
        for c_i in range(col):
            r, c = right_index[c_i, r_i]

            pic_name = "IMG{}x{}.tif".format(str(r).zfill(3), str(c).zfill(3))
            print(pic_name)
            img = np.asarray(Image.open(os.path.join(pics_dir, pic_name)))
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 旋转角度

            # show_img(img)
            whole_img[c_i * img.shape[0]: (c_i + 1) * img.shape[0], r_i * img.shape[1]: (r_i + 1) * img.shape[1]] = img

    # tifffile.imwrite(os.path.join(pics_dir, "whole_img.tif"), whole_img, compression="jpeg")
    return whole_img


def generate_whole_tif_gray(pics_dir, pixes=(2048, 2448)):
    # pics_dir = r"E:\biomarker_data\test_machine\Images"
    # pixes = np.array((2048, 2448))
    img_names = os.listdir(pics_dir)

    rows = [int(name[3:6]) for name in img_names if name.endswith(".tif")]
    cols = [int(name[7:10]) for name in img_names if name.endswith(".tif")]

    # 设置矩阵形状
    row, col = max(rows), max(cols)

    # 生成矩阵
    result_matrix = generate_matrix(row, col)

    # 顺时针偏转90°
    right_index = np.rot90(result_matrix, 3)  # 存着正确的序号的矩阵
    # 打印结果
    # print(right_index)
    # 下面开始拼接图像
    # de_pixes = pixes // 2
    de_pixes = pixes

    whole_size = (de_pixes[1] * col, de_pixes[0] * row)
    whole_img = np.zeros(whole_size, dtype=np.uint8)
    for r_i in range(row):
        for c_i in range(col):
            r, c = right_index[c_i, r_i]

            pic_name = "IMG{}x{}.tif".format(str(r).zfill(3), str(c).zfill(3))
            print(pic_name)
            img = tifffile.imread(os.path.join(pics_dir, pic_name))
            # img = np.asarray(Image.open(os.path.join(pics_dir, pic_name)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = img[..., 2]
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 旋转角度

            # show_img(img)
            whole_img[c_i * img.shape[0]: (c_i + 1) * img.shape[0], r_i * img.shape[1]: (r_i + 1) * img.shape[1]] = img

    return whole_img

if __name__ == '__main__':
    print(sys.argv)
    # sys.argv = ['.\\ChipDecodeScript.py', '-d', 'E:\\biomarker_data\\chip4_project', '--mrxs_dir', 'E:\\biomarker_data\\chip4']
    parser = argparse.ArgumentParser(description="merge HDX img_dirs")
    parser.add_argument('--save_dir', type=str, help="imgs save dir", default=None)
    parser.add_argument('--img_dir', nargs="+", type=str, help="all img dir path")
    args = parser.parse_args()

    print(args)

    save_dir = args.save_dir
    img_dir = args.img_dir
    print(img_dir)
    if len(img_dir)>3:
        raise Exception("路径不能大于3, imgs must < 3")

    if save_dir is None:
        save_dir = os.path.split(img_dir[0])[0]  # 默认为传入第一个路径的上一层

    whole_imgs = []
    for img_num, tmp_dir in enumerate(img_dir):

        whole_img = generate_whole_tif_gray(tmp_dir)
        whole_imgs.append(whole_img)
        tifffile.imwrite(os.path.join(save_dir, "img_{}.tif".format(img_num)), whole_img, compression="jpeg")

    final_img = np.zeros((whole_img.shape[0], whole_img.shape[1], 3), dtype=np.uint8)
    for i in range(len(whole_imgs)):
        final_img[..., i] = whole_imgs[i]

    tifffile.imwrite(os.path.join(save_dir, "merge_img.tif"), final_img, compression="jpeg")
    # img_dir = r"E:\biomarker_data\test_machine\Images"
    # whole_img = generate_whole_tif_RGB(img_dir)
    #
    # tifffile.imwrite(os.path.join(img_dir, "whole_img.tif"), whole_img, compression="lzw")