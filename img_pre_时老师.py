import os.path
import cv2
import tifffile
import numpy as np


def convert16_2_8(img_16):

    img_8bit = cv2.convertScaleAbs(img_16, alpha=(255.0/img_16.max()))
    # tifffile.imwrite(save_path, img_8bit, compression='LZW')
    # img_8bit = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2GRAY)
    img_8bit = cv2.convertScaleAbs(img_8bit, alpha=(255.0 / img_8bit.max()))
    return img_8bit.astype(np.uint8)


if __name__ == '__main__':
    img_type = "fl"
    image_path = r"E:\test\时老师项目\241212P26 -CY3.tif"
    save_dir = os.path.split(image_path)[0]
    img_name = os.path.split(image_path)[1]
    # 如果是荧光
    if img_type == "fl":
        chanel_1 = convert16_2_8(tifffile.imread(image_path, series=2))
        chanel_2 = convert16_2_8(tifffile.imread(image_path, series=1))
        chanel_3 = np.zeros_like(chanel_2)
        img = cv2.merge([chanel_1, chanel_2, chanel_3])
    # 如果是HE
    elif img_type == "he":
        img = convert16_2_8(tifffile.imread(image_path, series=1))

    img = np.rot90(img)  # 逆时针转90
    tifffile.imwrite(os.path.join(save_dir, img_name.replace(".tif", "_Trans.tif")), img, compression="lzw")
