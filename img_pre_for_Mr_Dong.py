import cv2
from PIL import Image
import os
import tifffile
import numpy as np
import argparse


def convert16_2_8(img_path, save_path=None):
    if save_path is None:
        save_path = os.path.splitext(img_path)[0] + "-8bit.tif"

    img_16 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # img_16 = tifffile.imread(img_path)
    if img_16.dtype == np.uint16:
        # img_16 = np.asarray(Image.open(img_path))
        img_8bit = cv2.convertScaleAbs(img_16, alpha=(255.0/img_16.max()))
        # tifffile.imwrite(save_path, img_8bit, compression='LZW')
        img_8bit = cv2.convertScaleAbs(img_8bit, alpha=(255.0 / img_8bit.max()))
    else:
        img_8bit = img_16
        img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY)
    return img_8bit.astype(np.uint8)

def merge_gray_to_RGB(imgs_dir):
    img_names = os.listdir(imgs_dir)
    if len(img_names) > 3 or len(img_names) == 0:
        raise Exception(f"Images num > 3 or ==0, check dir: {imgs_dir}")
    img_names.sort()

    img_0 = convert16_2_8(os.path.join(imgs_dir, img_names[0]))
    # img_0 = cv2.cvtColor(cv2.imread(os.path.join(imgs_dir, img_names[0])), cv2.COLOR_RGB2HSV)[..., -1]
    final_img = np.zeros((img_0.shape[0], img_0.shape[1], 3), dtype=np.uint8)
    final_img[..., 0] = img_0
    for i in range(1, len(img_names)):
        img_tmp = convert16_2_8(os.path.join(imgs_dir, img_names[i]))
        # img_tmp = cv2.cvtColor(tifffile.imread(os.path.join(imgs_dir, img_names[i])), cv2.COLOR_RGB2GRAY)
        # img_tmp = cv2.cvtColor(cv2.imread(os.path.join(imgs_dir, img_names[i])), cv2.COLOR_RGB2HSV)[..., -1]
        final_img[..., i] = img_tmp
    tifffile.imwrite(imgs_dir + ".tif", final_img, compression='LZW')
    return imgs_dir + ".tif"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="董老师扫描仪图像预处理，主要包含旋转和尺寸缩放")

    parser.add_argument('-P', type=int, nargs="+", help="pixes shape for single FOV. eg. -P 1536 2048 ", default=[1536, 2048])
    parser.add_argument('-S', type=int, nargs="+", help="FOV shape: row col. eg. -S 15 11", required=True)
    parser.add_argument('-I', type=str, help="Input Image path, or images dir", required=True)

    args = parser.parse_args()

    pixes = args.P
    FOV_shape = args.S
    img_path = args.I

    if not (len(pixes) == 2 and len(FOV_shape) == 2):
        raise Exception(f"Error input: {pixes}, {FOV_shape}")

    # 如果给的是图片文件夹，则需要将各个通道灰度图先合成一张RGB
    if os.path.isdir(img_path):
        img_path = merge_gray_to_RGB(img_path)

    new_size = np.asarray(pixes) * FOV_shape
    print(pixes)
    print(FOV_shape)
    print(new_size)

    # img_ori = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # img_ori = np.asarray(Image.open(img_path))
    img_ori = tifffile.imread(img_path)
    img_new = np.rot90(img_ori, -1)
    img_new = cv2.resize(img_new, new_size)

    save_path = os.path.splitext(img_path)[0] + "_new.tif"
    print(f"Save path: {save_path}")
    tifffile.imwrite(save_path, img_new, compression="LZW")
